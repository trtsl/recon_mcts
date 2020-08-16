//! A [`recon_mcts`](../recon_mcts/index.html) API demonstration of a simple
//! [Nim](https://en.wikipedia.org/wiki/Nim) variant. See
//! [code](../src/recon_mcts_test_nim/lib.rs.html#1-999).
//!
//! The game involves two players, a cumulative value (initially set to 500 in this example), and a
//! maximum move value (10 in this example).  The players take turns to reach 0 by subtracting a
//! move value between 1 and 10 from the cumulative value.  The first player to reach 0 wins.
//!
//! This game presents an interesting demonstration for tree search because it is intractable for a
//! non-recombining tree.  In the example given above, the search space for a non-recombining tree
//! is greater than 10<sup>50</sup>.
//!
//! # Implementation Note
//!
//! Note that this implementation is designed to be a demonstration of the API and therefore less
//! succinct than it could be.  For example, both static and dynamic dispatch version of the game
//! are implemented via [`GameDynamics`] and [`DynGD`], respectively.
//!
//! [`GameDynamics`]: ../recon_mcts/trait.GameDynamics.html
//! [`DynGD`]: ../recon_mcts/trait.DynGD.html

// A discussion of solving Nim via a DAG: https://webdocs.cs.ualberta.ca/~hayward/355/jem/nim.html

use recon_mcts::prelude::*;

use std::cell::Ref;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::Rng;

#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum Player {
    P1,
    P2,
}

// Each player keeps their own score ... just for fun
#[doc(hidden)]
#[derive(Debug)]
pub struct Score {
    player1: f64,
    player2: f64,
    visits_direct: AtomicUsize,
}

impl Score {
    fn fetch_add_visits_direct(&self, x: usize) -> usize {
        self.visits_direct.fetch_add(x, Ordering::Relaxed)
    }
}

impl Clone for Score {
    fn clone(&self) -> Self {
        let Score {
            ref player1,
            ref player2,
            ref visits_direct,
        } = self;
        Score {
            player1: *player1,
            player2: *player2,
            visits_direct: AtomicUsize::new(visits_direct.load(Ordering::Relaxed)),
        }
    }
}

#[doc(hidden)]
pub struct ActionIter {
    player: Player,
    nums: std::ops::RangeInclusive<usize>,
}

impl ActionIter {
    fn new(player: Player, max: usize) -> Self {
        Self {
            player,
            nums: 1..=max,
        }
    }
}

impl Iterator for ActionIter {
    type Item = (Player, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let a = self.nums.next()?;
        Some((self.player.clone(), a))
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct Nim {
    max_move: usize,
    // Include an rng to allow our implementation to include a random exploration incentive
    rng: Mutex<StdRng>,
}

// if static dispatch is not needed, then Nim can implement only `BaseGD` instead of `GameDynamics`
impl GameDynamics for Nim {
    type Player = Player;
    type State = usize;
    type Action = usize;
    type Score = Score;
    // If the returned iterator contains a closure, it will need to be boxed until
    // `#![feature(type_alias_impl_trait)]` is implemented (currently available in nightly):
    // https://github.com/rust-lang/rust/issues/63066
    // https://github.com/rust-lang/rust/issues/63063
    type ActionIter = ActionIter;

    fn available_actions(
        &self,
        player: &Self::Player,
        _state: &Self::State,
    ) -> Option<Self::ActionIter> {
        match player {
            Player::P1 => Some(ActionIter::new(Player::P2, self.max_move)),
            Player::P2 => Some(ActionIter::new(Player::P1, self.max_move)),
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        if state < *action {
            None
        } else {
            Some(state - action)
        }
    }

    fn select_node<II, Q, A>(
        &self,
        _parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        _parent_state: &Self::State,
        purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        // you can comment the line below and it'll still run (but it runs more slowly on some
        // machines ...  mysterious);  looking at the assembly one noticable difference was that
        // including the vector resulted in use of `mov[au]pd` instructions while excluding it
        // resulted in `mov[au]ps` instructions
        let scores_and_actions = scores_and_actions.into_iter().collect::<Vec<_>>();

        match parent_player {
            Player::P1 => scores_and_actions
                .into_iter()
                .map(|(q, a)| {
                    let qp = q.as_ref().unwrap().player1;
                    let e = match purpose {
                        SelectNodeState::Explore => self.rng.lock().unwrap().gen_range(-0.1, 0.1),
                        SelectNodeState::Exploit => 0.0,
                    };
                    (q, a, qp + e)
                })
                .max_by(|(.., a), (.., b)| a.partial_cmp(&b).unwrap())
                .map(|(q, a, _)| {
                    q.as_ref().unwrap().fetch_add_visits_direct(1);
                    a
                })
                .unwrap(),
            Player::P2 => scores_and_actions
                .into_iter()
                .map(|(q, a)| {
                    let qp = q.as_ref().unwrap().player2;
                    let e = match purpose {
                        SelectNodeState::Explore => self.rng.lock().unwrap().gen_range(-0.1, 0.1),
                        SelectNodeState::Exploit => 0.0,
                    };
                    (q, a, qp + e)
                })
                .max_by(|(.., a), (.., b)| a.partial_cmp(&b).unwrap())
                .map(|(q, a, _)| {
                    q.as_ref().unwrap().fetch_add_visits_direct(1);
                    a
                })
                .unwrap(),
        }
        .to_owned()
    }

    fn backprop_scores<II, Q>(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: IntoIterator<Item = Q>,
        Q: Deref<Target = Self::Score>,
    {
        let iter = child_scores.into_iter();
        let score = match player {
            Player::P1 => iter.max_by(|a, b| {
                let Score { player1: ref a, .. } = **a;
                let Score { player1: ref b, .. } = **b;
                a.partial_cmp(b).unwrap()
            }),
            Player::P2 => iter.max_by(|a, b| {
                let Score { player2: ref a, .. } = **a;
                let Score { player2: ref b, .. } = **b;
                a.partial_cmp(b).unwrap()
            }),
        }?
        .deref()
        .clone();

        // No need to propagate the score up the tree if it didn't change (recall the only scores
        // possible in this implementation are 0.0, 0.5, and 1.0).
        if let Some(Score {
            player1: ref s1,
            player2: ref s2,
            ref visits_direct,
            ..
        }) = score_current
        {
            if (s1 - score.player1).abs() + (s2 - score.player2).abs() < 0.1 {
                return None;
            }

            score
                .visits_direct
                .store(visits_direct.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        Some(score)
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        // std::thread::sleep(std::time::Duration::from_millis(1));
        Some(match parent_player {
            Player::P1 if *state == 0 => Score {
                player1: 1.0,
                player2: 0.0,
                visits_direct: AtomicUsize::new(1),
            },
            Player::P2 if *state == 0 => Score {
                player1: 0.0,
                player2: 1.0,
                visits_direct: AtomicUsize::new(1),
            },
            _ => Score {
                player1: 0.5,
                player2: 0.5,
                visits_direct: AtomicUsize::new(1),
            },
        })
    }
}

// Allow `Nim` to be used as a `GameDynamics` trait object via `DynGD` as long as all the
// associated types of `GameDynamics` match.  Doing so requires implementing the two methods of
// `GameDynamics` that have generic parameters for `DynGD` (`select_node` and `backprop_scores`).
// `DynGD` is a supertrait of `GameDynamics`.  Because of the special requirements on `DynGD`
// related to the use of a `RefCell` (see documentation), clippy throws a lot of irrelevant
// warnings, which we'll hide here.
#[allow(clippy::type_complexity)]
#[allow(clippy::clone_on_copy)]
#[allow(clippy::redundant_clone)]
#[allow(clippy::map_clone)]
impl DynGD for Nim {
    fn select_node(
        &self,
        _parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        _parent_node_state: &Self::State,
        purpose: SelectNodeState,
        scores_and_actions: &mut (dyn Iterator<
            Item = (Ref<'_, Option<Self::Score>>, Ref<'_, Self::Action>),
        >),
    ) -> Self::Action {
        // you can comment the line below and it'll still run (but it runs more slowly on some
        // machines ...  mysterious)
        let scores_and_actions = scores_and_actions
            .map(|(q, a)| (q.clone(), a.clone()))
            .collect::<Vec<_>>()
            .into_iter();

        // must `clone` item held by `Ref` from `scores_and_actions` before calling `max_by`
        // because `max_by` interleaves borrowing of the returned `Ref`s, which is not allowed by
        // the implemention, resulting in a runtime panic
        match parent_player {
            Player::P1 => scores_and_actions
                .map(|(q, a)| {
                    let qp = q.as_ref().unwrap().player1;
                    let e = match purpose {
                        SelectNodeState::Explore => self.rng.lock().unwrap().gen_range(-0.1, 0.1),
                        SelectNodeState::Exploit => 0.0,
                    };
                    (q.clone(), a.clone(), qp + e)
                })
                .max_by(|(.., a), (.., b)| a.partial_cmp(&b).unwrap()),
            Player::P2 => scores_and_actions
                .map(|(q, a)| {
                    let qp = q.as_ref().unwrap().player2;
                    let e = match purpose {
                        SelectNodeState::Explore => self.rng.lock().unwrap().gen_range(-0.1, 0.1),
                        SelectNodeState::Exploit => 0.0,
                    };
                    (q.clone(), a.clone(), qp + e)
                })
                .max_by(|(.., a), (.., b)| a.partial_cmp(&b).unwrap()),
        }
        .map(|(q, a, _)| {
            q.as_ref().unwrap().fetch_add_visits_direct(1);
            a
        })
        .unwrap()
    }

    fn backprop_scores(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores: &mut (dyn Iterator<Item = Ref<'_, Self::Score>>),
    ) -> Option<Self::Score> {
        // must `clone` item held by `Ref` from `child_scores` before calling `max_by` because
        // `max_by` interleaves borrowing of the returned `Ref`s, which is not allowed by the
        // implemention, resulting in a runtime panic
        let score: Self::Score = match player {
            Player::P1 => child_scores.map(|q| q.clone()).max_by(|a, b| {
                let Score { player1: ref a, .. } = a;
                let Score { player1: ref b, .. } = b;
                a.partial_cmp(b).unwrap()
            }),
            Player::P2 => child_scores.map(|q| q.clone()).max_by(|a, b| {
                let Score { player2: ref a, .. } = a;
                let Score { player2: ref b, .. } = b;
                a.partial_cmp(b).unwrap()
            }),
        }?;

        // No need to propagate the score up the tree if it didn't change (recall the only scores
        // possible in this implementation are 0.0, 0.5, and 1.0).
        if let Some(Score {
            player1: ref s1,
            player2: ref s2,
            ref visits_direct,
            ..
        }) = score_current
        {
            if (s1 - score.player1).abs() + (s2 - score.player2).abs() < 0.1 {
                return None;
            }

            score
                .visits_direct
                .store(visits_direct.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        Some(score)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use std::sync::Arc;

    const INIT: usize = 500;
    const MAX_MOVE: usize = 10;

    #[test]
    fn test_tree_static() {
        for test_f in &[run_single_thread, run_multi_thread] {
            let game = Nim {
                max_move: MAX_MOVE,
                rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(0)),
            };

            let t = Tree::new(game, GetState, Player::P1, INIT);

            print!("\nStatic");
            test_f(t);
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_tree_dynamic() {
        for test_f in &[run_single_thread, run_multi_thread] {
            let game = Nim {
                max_move: MAX_MOVE,
                rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(0)),
            };

            let game: Box<
                dyn 'static
                    + Send
                    + Sync
                    + DynGD<
                        Player = <Nim as BaseGD>::Player,
                        State = <Nim as BaseGD>::State,
                        Action = <Nim as BaseGD>::Action,
                        Score = <Nim as BaseGD>::Score,
                        ActionIter = <Nim as BaseGD>::ActionIter,
                    >,
            > = Box::new(game);

            let t = Tree::new(game, GetState, Player::P1, INIT);

            print!("\nDynamic");
            test_f(t);
        }
    }

    // Note that this single threaded implementation makes some moves before the winning move has
    // been determined (for demonstration purposes)
    fn run_single_thread<T, GD>(t: T)
    where
        T: SearchTree<GD = GD, Memory = GetState>,
        GD: GameDynamics,
        <GD as GameDynamics>::State: Debug + Clone + PartialEq + Hash,
        <GD as GameDynamics>::Action: Debug + Clone + Eq + Hash,
        <GD as GameDynamics>::Player: Debug + Clone + PartialEq + Hash,
        <GD as GameDynamics>::Score: Debug + Clone,
    {
        println!(" single thread");

        let t0 = std::time::Instant::now();
        // make 100 moves
        for _ii in 0.. {
            // for each move, expand the tree 100 times
            for _jj in 0..100 {
                // println!("{}:{} best action so far: {:?}", _ii, _jj, t.best_action());
                if t.step().is_none() {
                    break;
                }
            }

            match t.apply_best_action() {
                Status::Action(a) => {
                    println!("best action: {:?}", a);
                    println!("root state: {:?}", t.get_root_info());
                }
                Status::Pending | Status::ActionWip(_) => unreachable!(),
                Status::Terminal => break,
            }

            let children = t.find_children_sorted_with_depth();
            println!(
                "nnodes: {} depth: {}",
                children.len(),
                children.last().unwrap().1
            );
            assert_eq!(t.get_registry_nodes().len(), children.len());
        }
        println!("Elapsed: {:?}", t0.elapsed());
    }

    fn search_done<T, GD, M>(t: &T) -> bool
    where
        T: SearchTree<GD = GD, Memory = M>,
        GD: GameDynamics<Score = Score, Player = Player>,
    {
        match t.get_root_info() {
            // done if no moves left
            NodeInfo {
                n_children: Status::Terminal,
                ..
            } => true,
            // done if game has been solved
            NodeInfo {
                score:
                    Some(Score {
                        player1: ref p1,
                        player2: ref p2,
                        ..
                    }),
                ..
            } if *p1 > 0.8 || *p2 > 0.8 => true,
            // otherwise not done
            _ => false,
        }
    }

    fn run_multi_thread<T, GD>(t: T)
    where
        T: SearchTree<GD = GD, Memory = GetState> + Send + Sync + 'static,
        GD: GameDynamics<
            Player = <Nim as BaseGD>::Player,
            State = <Nim as BaseGD>::State,
            Action = <Nim as BaseGD>::Action,
            Score = <Nim as BaseGD>::Score,
            ActionIter = <Nim as BaseGD>::ActionIter,
        >,
    {
        println!(" multi thread");

        let t = Arc::new(t);

        let t0 = std::time::Instant::now();
        let handles = (0..4)
            .map(|_| {
                std::thread::spawn({
                    let t = Arc::clone(&t);
                    move || {
                        let t0 = std::time::Instant::now();
                        while !search_done(&*t) {
                            let _r = t.step();
                            // println!("step result: {:?}", _r);
                        }
                        println!("Elapsed: {:?}", t0.elapsed());
                    }
                })
            })
            .collect::<Vec<_>>();

        handles
            .into_iter()
            .map(std::thread::JoinHandle::join)
            .for_each(std::result::Result::unwrap);

        let (v, _) = recon_mcts::prelude::test_depth_helper(t.get_tree());
        let max_depth_field = v
            .iter()
            .map(|(n, _)| {
                let depth = n.get_node_info().depth;
                let state = get_state(n);
                assert!(depth + state <= INIT);
                depth
            })
            .max()
            .unwrap();

        assert_eq!(max_depth_field, v.last().unwrap().1);

        while let Status::Action(a) | Status::ActionWip(a) = t.apply_best_action() {
            let node_info = t.get_root_info();
            println!("best action: {:?}", a);
            println!("root state: {:?}", node_info);

            // `node_info` is called after `apply_best_action` so `node_info.player` is the next
            // player after action `a` is applied
            if node_info.player == Player::P2 && node_info.score.as_ref().unwrap().player1 > 0.8
                || node_info.player == Player::P1 && node_info.score.as_ref().unwrap().player2 > 0.8
            {
                assert_eq!(0, node_info.state.unwrap() % (MAX_MOVE + 1));
            }
        }
        println!("Elapsed total: {:?}", t0.elapsed());
    }
}
