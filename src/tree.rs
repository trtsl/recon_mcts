#![allow(clippy::type_complexity)]

use crate::game_dynamics::{GameDynamics, SelectNodeState};
use crate::lockref;
use crate::unique_heap::{self, UniqueHeap};

use std::cmp::Reverse;
use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};

pub type TreeAlias<GD, M> = Tree<NodeAlias<GD, M>, GD>;

pub type NodeAlias<GD, M> = Node<
    GD,
    <GD as GameDynamics>::State,
    <GD as GameDynamics>::Player,
    <GD as GameDynamics>::Action,
    <GD as GameDynamics>::Score,
    <<GD as GameDynamics>::ActionIter as IntoIterator>::IntoIter,
    M,
>;

/// An interface to reduce the number of bounds required to use a [`Tree`](./struct.Tree.html)
/// generically (i.e. the `SearchTree` trait is used to avoid having to list the bounds used to
/// [`implement`](trait.SearchTree.html#implementors) `SearchTree` for `Tree`).
///
/// In the descriptions below the terms `Tree` and `SearchTree` maybe be used interchangeably.
///
/// # Examples
///
/// ```
/// use recon_mcts::prelude::*;
///
/// fn use_tree_generically<T, GD, M>(t: &T) -> Status<<GD as GameDynamics>::Action>
/// where
///     T: SearchTree<GD = GD, Memory = M>,
///     GD: ?Sized + GameDynamics,
///     M: ?Sized,
/// {
///     while let Some(_) = t.step() {
///         // ...
///     }
///     t.apply_best_action()
/// }
///
/// fn use_tree_generically_more<T, GD, M>(t: &T)
/// where
///     T: SearchTree<GD = GD, Memory = M>,
///     GD: ?Sized + GameDynamics,
///     M: ?Sized,
///     <GD as GameDynamics>::Player: Clone,
///     <GD as GameDynamics>::Score: Clone,
/// {
///     use_tree_generically(t);
///     let info = t.get_root_info();
///     let moves = t.get_next_move_info();
///     // ...
/// }
///
/// fn use_tree_generically_more_other<T, GD, M>(t: &T)
/// where
///     T: SearchTree<GD = GD, Memory = M>,
///     GD: ?Sized + GameDynamics,
///     M: ?Sized,
///     NodeAlias<GD, M>: OnDrop,
/// {
///     use_tree_generically(t);
///     let children = t.find_children_sorted_with_depth();
///     let nodes = t.get_registry_info();
///     // ...
/// }
/// ```
pub trait SearchTree {
    /// A type that implements [`GameDynamics`](./trait.GameDynamics.html).
    type GD: ?Sized + GameDynamics;

    /// A [`state_memory`](./state_memory/index.html) mixin type used to configure how a `Node`'s
    /// state is stored.
    type Memory: ?Sized;

    /// Performs one iteration to expand the `SearchTree`.  Returns `Some(state)` if the
    /// `SearchTree` was expanded with a new leaf node and `None` otherwise.  The `state` in
    /// `Some(state)` is the `GameDynamics::State` of the `Node` that was expanded.  Note that in a
    /// multi-threaded context it is possible for this method to return `None` even though
    /// subsequent calls return `Some(_)`.  If this method is employed by the user to determine
    /// whether progress has been made, it is the user's responsibility to check that no other
    /// threads expanded the `SearchTree` during the execution of this method (`SearchTree`
    /// expansion results in an update of scores in the `SearchTree`, which could lead to the
    /// exploration of a new area of the `SearchTree`).
    fn step(&self) -> Option<<Self::GD as GameDynamics>::State>;

    /// Returns a `Status` with the currently anticipated `GameDynamics::Action` if available.
    fn best_action(&self) -> Status<<Self::GD as GameDynamics>::Action>;

    /// Move the root based on the selected action
    fn apply_action(&self, a: &<Self::GD as GameDynamics>::Action);

    /// Check for the best action and then apply it to move the root
    fn apply_best_action(&self) -> Status<<Self::GD as GameDynamics>::Action>;

    /// Returns a `NodeInfo` for the `SearchTree`'s root.
    fn get_root_info(
        &self,
    ) -> NodeInfo<
        <Self::GD as GameDynamics>::State,
        <Self::GD as GameDynamics>::Player,
        <Self::GD as GameDynamics>::Score,
    >
    where
        <Self::GD as GameDynamics>::Player: Clone,
        <Self::GD as GameDynamics>::Score: Clone;

    /// Returns `Some(Vec<(GameDynamics::Action, NodeInfo)>)` of all possible
    /// `GameDynamics::Action`s available from the `SearchTree`'s root.  Returns a `None` if no
    /// actions are available or their existence has not been determined by calling
    /// [`SearchTree::step`](trait.SearchTree.html#tymethod.step).
    fn get_next_move_info(
        &self,
    ) -> Option<
        Vec<(
            <Self::GD as GameDynamics>::Action,
            NodeInfo<
                <Self::GD as GameDynamics>::State,
                <Self::GD as GameDynamics>::Player,
                <Self::GD as GameDynamics>::Score,
            >,
        )>,
    >
    where
        <Self::GD as GameDynamics>::Player: Clone,
        <Self::GD as GameDynamics>::Score: Clone;

    /// Returns a vector of topologically sorted `(ArcNode, usize)` pairs where the `usize`
    /// indicates the distance from the `ArcNode` to the leaf that has the maximum reachable depth.
    /// The vector is sorted such that index `0` is a leaf and the last element is the root node.
    /// The vector represents the children, grandchildren, etc. of the root node (as well as the
    /// root itself).  Note that `usize` is always strictly smaller for a child than its parent.
    /// This is basically a [depth first
    /// search](https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search).
    fn find_children_sorted_with_depth(
        &self,
    ) -> Vec<(ArcWrap<NodeAlias<Self::GD, Self::Memory>>, usize)>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop;

    /// Returns a `HashSet` of all `Node`s currently in the `SearchTree`.
    fn get_registry_nodes(&self) -> HashSet<WeakWrap<NodeAlias<Self::GD, Self::Memory>>>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop;

    /// Returns summary statistics for the `SearchTree`'s registry.
    fn get_registry_info(&self) -> &RegistryInfo;

    /// Returns a reference to the game dynamics.
    fn get_game_dynamics(&self) -> Arc<Self::GD>;

    #[doc(hidden)]
    #[cfg(any(test, feature = "test_internals"))]
    fn get_tree(&self) -> &TreeAlias<Self::GD, Self::Memory>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop;
}

impl<GD, S, P, A, Q, I, M, II> SearchTree for TreeAlias<GD, M>
where
    Node<GD, S, P, A, Q, I, M>: StateMemory<State = S>,
    GD: GameDynamics<Player = P, State = S, Action = A, Score = Q, ActionIter = II>,
    II: IntoIterator<IntoIter = I, Item = (P, A)>,
    I: Iterator<Item = (P, A)>,
    A: Clone + Hash + Eq,
    S: Clone + Hash + PartialEq<S>,
    P: Hash + PartialEq<P>,
{
    type GD = GD;
    type Memory = M;

    #[inline(always)]
    fn step(&self) -> Option<<Self::GD as GameDynamics>::State> {
        Self::step(self)
    }

    #[inline(always)]
    fn best_action(&self) -> Status<<Self::GD as GameDynamics>::Action> {
        Self::best_action(&self)
    }

    #[inline(always)]
    fn apply_action(&self, a: &<Self::GD as GameDynamics>::Action) {
        Self::apply_action(self, a)
    }

    #[inline(always)]
    fn apply_best_action(&self) -> Status<<Self::GD as GameDynamics>::Action> {
        Self::apply_best_action(self)
    }

    #[inline(always)]
    fn get_root_info(
        &self,
    ) -> NodeInfo<
        <Self::GD as GameDynamics>::State,
        <Self::GD as GameDynamics>::Player,
        <Self::GD as GameDynamics>::Score,
    >
    where
        <Self::GD as GameDynamics>::Player: Clone,
        <Self::GD as GameDynamics>::Score: Clone,
    {
        Self::get_root_info(self)
    }

    #[inline(always)]
    fn get_next_move_info(
        &self,
    ) -> Option<
        Vec<(
            <Self::GD as GameDynamics>::Action,
            NodeInfo<
                <Self::GD as GameDynamics>::State,
                <Self::GD as GameDynamics>::Player,
                <Self::GD as GameDynamics>::Score,
            >,
        )>,
    >
    where
        <Self::GD as GameDynamics>::Player: Clone,
        <Self::GD as GameDynamics>::Score: Clone,
    {
        Self::get_next_move_info(self)
    }

    #[inline(always)]
    fn find_children_sorted_with_depth(
        &self,
    ) -> Vec<(ArcWrap<NodeAlias<Self::GD, Self::Memory>>, usize)>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop,
    {
        Self::find_children_sorted_with_depth(self)
    }

    #[inline(always)]
    fn get_registry_nodes(&self) -> HashSet<WeakWrap<NodeAlias<Self::GD, Self::Memory>>>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop,
    {
        Self::get_registry_nodes(self)
    }

    #[inline(always)]
    fn get_registry_info(&self) -> &RegistryInfo {
        Self::get_registry_info(self)
    }

    #[inline(always)]
    fn get_game_dynamics(&self) -> Arc<GD> {
        Self::get_game_dynamics(self)
    }

    #[cfg(any(test, feature = "test_internals"))]
    #[inline(always)]
    fn get_tree(&self) -> &TreeAlias<Self::GD, Self::Memory>
    where
        NodeAlias<Self::GD, Self::Memory>: OnDrop,
    {
        self
    }
}

/// Provides information about the result of applying a `GameDynamics::Action` to a `Tree`.
#[derive(Debug, Clone)]
pub enum Status<T> {
    /// It is currently unknown whether the game is over or not.
    Pending,
    /// Actions are available, but not all branches have been created.
    ActionWip(T),
    /// Actions are available, and all branches have been created.
    Action(T),
    /// The game is over.
    Terminal,
}

impl<T> Status<T> {
    fn from_children<I, A, N>(c: &Children<I, A, N>, f: impl FnOnce(&HashMap<A, N>) -> T) -> Self {
        match c {
            Children::NewLeaf => Status::Pending,
            Children::BranchWip(h) => Status::ActionWip(f(h.scored_ref())),
            Children::Branch(h) => Status::Action(f(h)),
            Children::None => Status::Terminal,
        }
    }
}

/// Contains information about a specific `Node`.
#[derive(Debug, Clone)]
pub struct NodeInfo<S, P, Q> {
    pub depth: usize,
    pub state: Option<S>,
    pub player: P,
    pub score: Option<Q>,
    pub n_parents: usize,
    pub n_children: Status<usize>,
}

use state_memory::StateMemory;
pub mod state_memory {
    //! Provides mixins to statically configure how each node's state is stored in memory and
    //! compared for equality.
    //!
    //! The implementation relies on a hash table to determine whether the state of a newly created
    //! leaf node is identical to that of an existing node (effectively a [transposition
    //! table](https://en.wikipedia.org/wiki/Transposition_table)).  For the purpose of determining
    //! whether two game states are identical, both [`GameDynamics::State`] and
    //! [`GameDynamics::Player`] are considered.
    //!
    //! The provided mixins configure how hash table comparisons are executed.  A high-level
    //! summary of strengths of each configuration is shown in the table below.  However, the
    //! decision is best made by testing the configurations to see which works best.
    //!
    //! **_The following mixins are available_**:
    //!
    //! **[`GetState`]**:  store the state only for the root node and a hash for all other nodes;
    //! recompute the state for equality comparisons by traversing the tree from the root to the
    //! target node.  
    //! **[`HashOnly`]**:  store the state only for the root node and a hash for all other nodes;
    //! rely solely on the state's hash for equality comparisons.  
    //! **[`StoreState`]**:  store the state of all nodes.
    //!
    //! <table>
    //! <tr><th>Mixin / Strengths</th><th><center>Accuracy</th><th><center>Memory</th><th><center>Performance</th></tr>
    //! <tr><td><a href="struct.GetState.html">GetState</a></td><td><center>&#x2713;</td><td><center>&#x2713;</td><td><center></td></tr>
    //! <tr><td><a href="struct.HashOnly.html">HashOnly</a></td><td><center></td><td><center>&#x2713;</td><td><center>&#x2713;</td></tr>
    //! <tr><td><a href="struct.StoreState.html">StoreState</a></td><td><center>&#x2713;</td><td><center></td><td><center>&#x2713;</td></tr>
    //! </table>
    //!
    //! The mixins are used in constructing a [`Tree`] as follows:
    //!
    //! ```no_run
    //! # use recon_mcts::prelude::*;
    //! # use std::hash::Hash;
    //! #
    //! # fn make_tree<GD, S, P, A, Q, II, I>(game: GD, first_player: P, root_state: S)
    //! # where
    //! #     // Node<GD, S, P, A, Q, I, M>: StateMemory<State = S>,
    //! #     GD: GameDynamics<Player = P, State = S, Action = A, Score = Q, ActionIter = II>,
    //! #     II: IntoIterator<IntoIter = I, Item = (P, A)>,
    //! #     I: Iterator<Item = (P, A)>,
    //! #     A: Hash + Eq + Clone,
    //! #     S: Hash + PartialEq<S> + Clone,
    //! #     P: Hash + PartialEq<P>,
    //! # {
    //! let tree = Tree::new(game, GetState, first_player, root_state);
    //! # }
    //! ```
    //!
    //! [`GetState`]: struct.GetState.html
    //! [`HashOnly`]: struct.HashOnly.html
    //! [`StoreState`]: struct.StoreState.html
    //! [`Tree`]: ../struct.Tree.html
    //! [`GameDynamics::State`]: ../trait.GameDynamics.html#associatedtype.State
    //! [`GameDynamics::Player`]: ../trait.GameDynamics.html#associatedtype.Player

    use super::Node;
    use crate::game_dynamics::GameDynamics;

    use std::hash::Hash;
    use std::sync::RwLock;

    /// A trait used to modify how states are stored in the transposition table.  Generally for
    /// internal use.
    // If desired, `StateMemory` could easily be extended to make the saving of states conditional on
    // e.g. the tree depth at the time of node creation, the number of nodes in the registry or other
    // characteristics of the game state. Consider implementing the `StateMemory` using the Sealed
    // Trait Pattern:
    // https://rust-lang.github.io/api-guidelines/future-proofing.html#c-sealed
    pub trait StateMemory {
        type State;

        // `PartialEq` for `Node`
        fn eq(&self, rhs: &Self) -> bool;

        // set how the state should be stored after a child is created
        fn modify_state(state: &RwLock<Option<Self::State>>);
    }

    /// Memory usage is state dependent (could use lots of storage if states are large).
    pub struct StoreState;
    impl<P, S, A, I, GD, Q> StateMemory for Node<GD, S, P, A, Q, I, StoreState>
    where
        GD: GameDynamics<Player = P, State = S, Action = A>,
        A: Hash + Eq,
        S: Hash + PartialEq<S> + Clone,
        P: Hash + PartialEq<P>,
    {
        type State = S;

        fn eq(&self, rhs: &Self) -> bool {
            self.player == rhs.player && self.get_state() == rhs.get_state()
        }

        fn modify_state(_: &RwLock<Option<Self::State>>) {}
    }

    /// Slower performance but better memory efficiency for large states.
    pub struct GetState;
    impl<P, S, A, I, GD, Q> StateMemory for Node<GD, S, P, A, Q, I, GetState>
    where
        GD: GameDynamics<Player = P, State = S, Action = A>,
        A: Hash + Eq,
        S: Hash + PartialEq<S> + Clone,
        P: Hash + PartialEq<P>,
    {
        type State = S;

        fn eq(&self, rhs: &Self) -> bool {
            self.player == rhs.player && self.get_state() == rhs.get_state()
        }

        fn modify_state(state: &RwLock<Option<Self::State>>) {
            *state.write().unwrap() = None;
        }
    }

    /// There's a chance of hash collision, which would mean that a parent node connects to an
    /// incorrect child node.
    pub struct HashOnly;
    impl<P, S, A, I, GD, Q> StateMemory for Node<GD, S, P, A, Q, I, HashOnly>
    where
        GD: GameDynamics<Player = P, State = S, Action = A>,
        A: Hash + Eq,
        S: Hash + PartialEq<S> + Clone,
        P: Hash + PartialEq<P>,
    {
        type State = S;

        fn eq(&self, rhs: &Self) -> bool {
            self.hash == rhs.hash
        }

        fn modify_state(state: &RwLock<Option<Self::State>>) {
            *state.write().unwrap() = None;
        }
    }
}

// NewLeaf: a child node without children of its own
// BranchWip: a node whose children have *not* all been scored via `GD::score_leaf`
// Branch: a node whose children have all been scored via `GD::score_leaf`
// None: a terminal node in the game that will never have children
enum Children<I, A, N> {
    NewLeaf,
    BranchWip(BranchWip<I, A, N>),
    Branch(HashMap<A, N>),
    None,
}

impl<I, A, N> Children<I, A, N> {
    fn as_map(&self) -> Option<&HashMap<A, N>> {
        match self {
            Children::BranchWip(h) => Some(h.scored_ref()),
            Children::Branch(h) => Some(h),
            _ => None,
        }
    }

    fn as_map_mut(&mut self) -> Option<&mut HashMap<A, N>> {
        match self {
            Children::BranchWip(h) => Some(h.scored_mut()),
            Children::Branch(h) => Some(h),
            _ => None,
        }
    }

    fn as_wip_mut(&mut self) -> Option<&mut BranchWip<I, A, N>> {
        match self {
            Children::BranchWip(b) => Some(b),
            _ => None,
        }
    }
}

use branch_wip::BranchWip;
mod branch_wip {
    // The main purpose is to provide an abstraction around building a new branch and provide a
    // `Condvar` such that other threads can check on the status of the branch and wait on its
    // completion (if they are not able to steal any of the work) without holding a reference to
    // `BranchWip` (which would require maintaining a lock)

    use std::cmp::Eq;
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

    pub(crate) struct Notifier {
        cv: Condvar,
        mtx: Mutex<bool>,
    }

    #[allow(clippy::mutex_atomic)]
    impl Notifier {
        fn new() -> Self {
            Self {
                cv: Condvar::new(),
                mtx: Mutex::new(false),
            }
        }

        pub fn notify_all(&self) -> bool {
            let prior = std::mem::replace(&mut *self.mtx.lock().unwrap(), true);
            self.cv.notify_all();
            prior
        }

        pub fn wait(&self) -> Result<MutexGuard<bool>, PoisonError<MutexGuard<bool>>> {
            let lk = self.mtx.lock().unwrap();
            self.cv.wait_while(lk, |n| !*n)
        }
    }

    pub(crate) struct BranchWip<I, A, N> {
        unscored: I,
        unscored_done: bool,
        scored: Option<HashMap<A, N>>,
        scores_pending: usize,
        notifier: Arc<Notifier>,
    }

    impl<I, A, N> BranchWip<I, A, N> {
        pub fn new(unscored: I) -> Self {
            Self {
                unscored,
                unscored_done: false,
                scored: Some(HashMap::new()),
                scores_pending: 0,
                notifier: Arc::new(Notifier::new()),
            }
        }

        pub fn scored_ref(&self) -> &HashMap<A, N> {
            self.scored.as_ref().unwrap()
        }

        pub fn scored_mut(&mut self) -> &mut HashMap<A, N> {
            self.scored.as_mut().unwrap()
        }

        pub fn next_unscored<P>(&mut self) -> Option<(P, A)>
        where
            I: Iterator<Item = (P, A)>,
        {
            let n = self.unscored.next();
            if n.is_some() {
                self.scores_pending += 1;
            } else {
                self.unscored_done = true;
            }
            n
        }

        pub fn scored_insert(&mut self, a: A, n: N)
        where
            A: Eq + Hash,
        {
            self.decrease_scores_pending();
            let _r = self.scored.as_mut().unwrap().insert(a, n);

            #[cfg(debug_assertions)]
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static SEEN: AtomicBool = AtomicBool::new(false);
                if !(_r.is_none() || SEEN.load(Ordering::Relaxed)) {
                    eprintln!(
                        "Warning: \
                        GameDynamics::ActionIter returned the same player/action pair twice\
                        "
                    );
                    SEEN.store(true, Ordering::Relaxed);
                }
            }
        }

        pub fn decrease_scores_pending(&mut self) {
            debug_assert!(self.scores_pending > 0);
            self.scores_pending -= 1;
        }

        pub fn take_scored(&mut self) -> Option<HashMap<A, N>> {
            self.scored.take()
        }

        pub fn get_notifier(&self) -> Arc<Notifier> {
            Arc::clone(&self.notifier)
        }

        pub fn finished(&self) -> bool {
            // checking this condition is sufficient as long as `next_unscored` is always called
            // before `scores_pending` such that `scores_pending == 0` implies the iterator is
            // exhausted
            self.unscored_done && self.scores_pending == 0
        }
    }
}

// GD = GameDynamics
// S  = GameDynamics::State
// P  = GameDynamics::Player
// A  = GameDynamics::Action
// Q  = GameDynamics::Score
// I  = Iterator<Item = (P, A)>,
// M  = StateMemory
/// The fundamental type composing a `Tree`.
pub struct Node<GD, S, P, A, Q, I, M>
where
    Self: OnDrop,
    GD: ?Sized,
    M: ?Sized,
{
    hash: u64,
    player: P,
    depth: AtomicUsize,
    state: RwLock<Option<S>>,
    score: RwLock<Option<Q>>,
    score_gen: AtomicUsize,
    parents: RwLock<HashSet<(A, WeakWrap<Self>)>>,
    children: RwLock<Children<I, A, ArcWrap<Self>>>,
    registry: Arc<RwLock<HashSet<WeakWrap<Self>>>>,
    registered: AtomicBool,
    game_dynamics: Arc<GD>,
    // Use `fn() -> M` in `PhantomData` because it is covariant over `M` like `M` itself (which
    // requires drop check because it suggests ownership) or `*const  M` (which is not `Send`);
    // though just using `M` and going through the drop check would really be ok here since `M` is
    // only used for compile time dispatch and no lifetimes are associated with it; see:
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#when-does-one-type-own-another
    _marker: PhantomData<fn() -> M>,
}

impl<GD, S, P, A, Q, I, M> Node<GD, S, P, A, Q, I, M>
where
    Self: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    fn new_root(
        game_dynamics: Arc<GD>,
        player: P,
        state: S,
        registry: Arc<RwLock<HashSet<WeakNode<GD, S, P, A, Q, I, M>>>>,
    ) -> ArcWrap<Self> {
        let node = Self {
            hash: Self::hash(&player, &state),
            player,
            depth: AtomicUsize::new(0),
            state: RwLock::new(Some(state)),
            score: RwLock::new(None),
            score_gen: AtomicUsize::new(0),
            parents: RwLock::new(HashSet::new()),
            children: RwLock::new(Children::NewLeaf),
            registry,
            registered: AtomicBool::new(false),
            game_dynamics,
            _marker: PhantomData,
        };
        let this = ArcWrap::<Self> {
            inner: Arc::new(node),
        };
        Self::register(&this, Option::<&mut &mut HashSet<_, _>>::None);
        this
    }

    fn new_child(
        parent_node: &ArcNode<GD, S, P, A, Q, I, M>,
        player: P,
        state: S,
    ) -> ArcWrap<Self> {
        // the depth must be set after the node is connected to its parents using
        // `Node::connect_child` but before the scores of the node are propagated using
        // `Node::backprop_scores`
        let depth = AtomicUsize::new(0);
        let registry = Arc::clone(&parent_node.registry);
        let game_dynamics = Arc::clone(&parent_node.game_dynamics);
        let hash = Node::<GD, S, P, A, Q, I, M>::hash(&player, &state);
        ArcNode {
            inner: Arc::new(Node {
                hash,
                player,
                depth,
                state: RwLock::new(Some(state)),
                score: RwLock::new(None),
                score_gen: AtomicUsize::new(0),
                parents: RwLock::new(HashSet::new()),
                children: RwLock::new(Children::NewLeaf),
                registry,
                registered: AtomicBool::new(false),
                game_dynamics,
                _marker: PhantomData,
            }),
        }
    }

    fn register<R>(self_arc: &ArcNode<GD, S, P, A, Q, I, M>, reg_wlk: Option<&mut R>)
    where
        R: DerefMut<Target = HashSet<WeakNode<GD, S, P, A, Q, I, M>>>,
    {
        let mut _r = match reg_wlk {
            Some(reg_wlk) => reg_wlk.insert(ArcNode::downgrade(self_arc)),
            None => self_arc
                .registry
                .write()
                .unwrap()
                .insert(ArcNode::downgrade(self_arc)),
        };

        _r &= !self_arc
            .registered
            .compare_and_swap(false, true, Ordering::Relaxed);

        debug_assert!(_r, "node already in registry");
    }

    fn connect_child(self_arc: &ArcWrap<Self>, a: A, child: &ArcNode<GD, S, P, A, Q, I, M>)
    where
        A: Clone,
    {
        // connection from child to self (parent)
        let _r = child
            .parents
            .write()
            .unwrap()
            .insert((a.clone(), ArcNode::downgrade(&self_arc)));

        // connection from self (parent) to child
        let mut children_wlk = self_arc.children.write().unwrap();
        let branch_wip_mut = children_wlk.as_wip_mut().unwrap();
        branch_wip_mut.scored_insert(a, ArcNode::clone(child));

        #[cfg(debug_assertions)]
        {
            static SEEN: AtomicBool = AtomicBool::new(false);
            if !(_r || SEEN.load(Ordering::Relaxed)) {
                eprintln!(
                    "Warning: \
                    GameDynamics::ActionIter returned the same player/action pair twice\
                    "
                );
                SEEN.store(true, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn get_state(&self) -> S {
        // for `GetState`, equality is ultimately determined by applying actions to the root state
        // and determining whether the final states are identical
        match &*self.state.read().unwrap() {
            Some(ref s) => s.clone(),
            None => {
                let parents = self.parents.read().unwrap();
                let (a, p) = parents
                    .iter()
                    .next()
                    .expect("can't calculate state for node without parents");
                let state = WeakNode::upgrade(&p).get_state();
                GD::apply_action(&*self.game_dynamics, state, a).unwrap()
            }
        }
    }

    fn move_root(&self, action: &A) -> ArcNode<GD, S, P, A, Q, I, M> {
        debug_assert_eq!(self.parents.read().unwrap().len(), 0);

        let new_root = self
            .children
            .write()
            .unwrap()
            .as_map_mut()
            .expect("root's children not (yet) a `Branch`")
            .remove(action)
            .expect("missing child for action");

        let r = new_root
            .parents
            .write()
            .unwrap()
            .drain()
            .map(|(a, wn)| (a, WeakNode::upgrade(&wn)))
            .filter(|(_, p)| p.as_ptr() != self.as_ptr())
            .all(|(a, p)| {
                p.children
                    .write()
                    .unwrap()
                    .as_map_mut()
                    .unwrap()
                    .remove(&a)
                    .is_some()
            });

        debug_assert!(r, "parent did not know about child");

        if let ref mut s @ None = *new_root.state.write().unwrap() {
            let state_old = self
                .state
                .write()
                .unwrap()
                .as_ref()
                .expect("move_root called from child")
                .clone();

            *s = GD::apply_action(&*self.game_dynamics, state_old, action);
        }

        new_root.parents.write().unwrap().clear();
        new_root
    }

    fn update_score(&self) -> bool
    where
        GD: GameDynamics<Score = Q>,
    {
        let children_rlk = self.children.read().unwrap();
        match *children_rlk {
            Children::Branch(ref map) => loop {
                // The loop ensures that an update to the score of a node's children is reflected
                // in the score of the node before the function returns, this in conjunction with
                // calling `Node::backprop_scores` on every new branch ensures all score updates
                // are eventually reflected in the root node's score.  The alternative is to
                // acquire `score_wlk` before running `GD::backprop_scores` but performance is less
                // optimal (`GD::backprop_scores` is implemented by the user and could be slow, and
                // a new leaf could require `update_score` to be called on *all* nodes in the
                // `Tree`, so there could be significant contention with other threads trying to
                // read scores in e.g. `GD::select_node`), or to include `score_gen` in a struct
                // along with the `score` though this also has slightly worse performance and
                // requires additional syntax clutter to destructure the score struct.  The atomic
                // `Ordering` enforces ordering for the relevant non-atomic data as well.  See:
                // https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
                // http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync
                let scores = map.iter().map(|(_, c)| {
                    lockref::Ref::new(c.score.read().unwrap(), |s| s.as_ref().expect("no score"))
                });
                // `Ordering::Acquire` because `GD::backprop_scores` should not be reordered before
                // loading `score_gen`
                let gen = self.score_gen.load(Ordering::Acquire);
                let score_cur_rlk = self.score.read().expect("no score");
                let score_new = GD::backprop_scores(
                    &*self.game_dynamics,
                    &self.player,
                    score_cur_rlk.as_ref(),
                    scores,
                );
                drop(score_cur_rlk);
                if let Some(score) = score_new {
                    let mut score_wlk = self.score.write().expect("no score");
                    // `Ordering::Release` because `GD::backprop_scores` should not be reordered after
                    // storing `score_gen`
                    let gen_prev = self
                        .score_gen
                        .compare_and_swap(gen, gen + 1, Ordering::Release);
                    if gen_prev == gen {
                        *score_wlk = Some(score);
                        break true;
                    }
                } else {
                    break false;
                }
            },
            // `self` can contain a `BranchWip` because `Tree::create_scored_child` may connect to
            // an existing node (i.e. one that is in the `registry`) on which `Node::update_score`
            // will be called before `self` is converted to a `Branch`; we can either wait on the
            // `BranchWip` to become a `Branch` (as below) or just skip over this node since we
            // know `Node::backprop_scores` will be called when it is converted to a `Branch`
            // Children::BranchWip(ref wip) => {
            //     let notifier = wip.get_notifier();
            //     drop(children_rlk);
            //     let _ = notifier.wait().unwrap();
            //     self.update_score();
            // }
            _ => false,
        }
    }

    fn backprop_scores(self_arc: &ArcWrap<Self>) -> usize
    where
        GD: GameDynamics<Score = Q>,
    {
        let mut n_updates = 0;
        let mut h = UniqueHeap::new();
        let d = self_arc.depth.load(Ordering::Relaxed);
        h.push((d, ArcWrap::clone(self_arc)));

        while let Some((_, node)) = h.pop() {
            if node.update_score() {
                n_updates += 1;

                node.parents.read().unwrap().iter().for_each(|(_, p)| {
                    let p = WeakWrap::upgrade(&p);
                    let dp = p.depth.load(Ordering::Relaxed);
                    h.push((dp, p));
                });
            }
        }

        n_updates
    }

    fn apply_atomic(
        base: &AtomicUsize,
        f_desired: &mut impl FnMut(usize) -> usize,
        order: Ordering,
    ) -> usize {
        // somewhat generic idea from https://software.intel.com/en-us/node/506125
        let mut cur = base.load(order);
        let mut desired = f_desired(cur);
        while cur != desired {
            let act = base.compare_and_swap(cur, desired, order);
            if act == cur {
                return desired;
            } else {
                cur = act;
                desired = f_desired(cur);
            }
        }
        desired
    }

    fn update_depth(&self) -> Option<usize> {
        let parents_rlk = self.parents.read().unwrap();

        let min_depth = 1 + parents_rlk
            .iter()
            .map(|(_, w)| WeakNode::upgrade(w).depth.load(Ordering::Relaxed))
            .max()?;

        let new_depth = Self::apply_atomic(
            &self.depth,
            &mut |x| std::cmp::max(x, min_depth),
            Ordering::Relaxed,
        );

        Some(new_depth)
    }

    fn set_min_depth(self_arc: &ArcWrap<Self>) -> usize {
        let mut n_updates = 0;
        let mut h = UniqueHeap::new();

        let d = self_arc.depth.load(Ordering::Relaxed);
        let elem = Reverse((d, ArcWrap::clone(self_arc)));
        h.push(elem);

        while let Some(Reverse((_, node))) = h.pop() {
            let old_depth = node.depth.load(Ordering::Relaxed);
            let new_depth = node.update_depth();
            match new_depth {
                Some(d) if d > old_depth => {}
                _ => continue,
            }

            n_updates += 1;

            let children_rlk = node.children.read().unwrap();
            if let Some(map) = children_rlk.as_map() {
                map.iter().for_each(|(_, c)| {
                    let c = ArcNode::clone(&c);
                    let d = c.depth.load(Ordering::Relaxed);
                    h.push(Reverse((d, c)));
                });
            }
        }

        n_updates
    }

    // Returns a vector of topologically sorted `ArcNode`s such that `sorted[0]` is the root node
    // and the last element is `self`.  The vector represents the parents, grandparents, etc. of
    // `self` (as well as `self` itself).  This is basically the following algorithm:
    // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    #[allow(dead_code)]
    pub(crate) fn find_parents_sorted(
        self_arc: &ArcWrap<Self>,
        sorted: &mut Vec<ArcWrap<Self>>,
        visited: &mut HashSet<*const Node<GD, S, P, A, Q, I, M>>,
    ) {
        if visited.insert(self_arc.as_ptr()) {
            self_arc.parents.read().unwrap().iter().for_each(|(_, p)| {
                Self::find_parents_sorted(&WeakNode::upgrade(&p), sorted, visited)
            });
            sorted.push(ArcNode::clone(self_arc));
        }
    }

    // Returns a vector of topologically sorted `ArcNode`s such that `sorted[0]` is a leaf node
    // with `inv_depth` == 0 and the last element is `self`.  The vector represents the children,
    // grandchildren, etc. of `self` (as well as `self` itself).  Note that `inv_depth` is always
    // smaller for a child compared to its parent.  This is basically the following algorithm:
    // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    pub(crate) fn find_children_sorted_with_depth(
        self_arc: &ArcWrap<Self>,
        sorted: &mut Vec<(ArcWrap<Self>, usize)>,
        visited: &mut HashMap<*const Node<GD, S, P, A, Q, I, M>, usize>,
    ) -> usize {
        match visited.get(&self_arc.as_ptr()) {
            Some(d) => *d,
            None => {
                let mut inv_depth = 0;
                if let Some(ref children) = self_arc.children.read().unwrap().as_map() {
                    inv_depth = 1 + children
                        .iter()
                        .map(|(_, c)| Self::find_children_sorted_with_depth(&c, sorted, visited))
                        .max()
                        .unwrap_or(0);
                }
                let _r = visited.insert(self_arc.as_ptr(), inv_depth);
                debug_assert!(_r.is_none(), "not a DAG");
                sorted.push((ArcNode::clone(&self_arc), inv_depth));
                inv_depth
            }
        }
    }

    fn as_ptr(&self) -> *const Self {
        self as *const _
    }

    fn hash(player: &P, state: &S) -> u64 {
        let mut hasher = DefaultHasher::new();
        player.hash(&mut hasher);
        state.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_node_info(&self) -> NodeInfo<S, P, Q>
    where
        Q: Clone,
        P: Clone,
    {
        NodeInfo {
            depth: self.depth.load(Ordering::Relaxed),
            player: self.player.clone(),
            score: self.score.read().unwrap().clone(),
            state: self.state.read().unwrap().clone(),
            n_parents: self.parents.read().unwrap().len(),
            n_children: Status::from_children(&*self.children.read().unwrap(), HashMap::len),
        }
    }
}

/// A trait used to remove nodes from the transposition table that are no longer reachable from the
/// root. Generally for internal use.
pub trait OnDrop {
    // TODO: once stabilized, turn `self_arc` to `self` using `#![feature(arbitrary_self_types)]`;
    // besides being being semantically more reflective of the intention, it will also make using
    // `GameDynamics` as a trait object easier because `OnDrop` currently requires types that
    // implement it to be `Sized` because of the `Self` parameter in accordance with object safety
    // rules https://doc.rust-lang.org/unstable-book/language-features/arbitrary-self-types.html
    // https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
    fn on_drop(self_arc: ArcWrap<Self>);
}

impl<GD, S, P, A, Q, I, M> OnDrop for Node<GD, S, P, A, Q, I, M>
where
    Self: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    fn on_drop(self_arc: ArcWrap<Self>) {
        if let Some(ref children) = self_arc.children.read().unwrap().as_map() {
            if !children.is_empty() && self_arc.state.read().unwrap().is_none() {
                // the orphan must have a state because it is needed when the orphan's children
                // remove the orphan as a parent; `Node::get_state` checks the node's state, so we
                // have to check the status of the node's state with only a read lock first, then,
                // if needed, calculate the state, and then write the state to the node
                let state = self_arc.get_state();
                *self_arc.state.write().unwrap() = Some(state);
            }
        }

        for (a, p) in self_arc.parents.read().unwrap().iter() {
            let r = p
                .upgrade()
                .children
                .write()
                .unwrap()
                .as_map_mut()
                .unwrap()
                .remove(a);

            debug_assert!(
                r.is_some(),
                "could not remove dropped node from parent's children"
            );
        }

        if let Some(ref mut children) = self_arc.children.write().unwrap().as_map_mut() {
            for (a, c) in children.drain() {
                // The below is effectively the same condition as:
                // if c.parents.read().unwrap().len() == 1 {
                if Arc::strong_count(&c.inner) == 1 {
                    *c.state.write().unwrap() = Some(c.inner.get_state());
                }

                let r = c
                    .parents
                    .write()
                    .unwrap()
                    .remove(&(a, ArcNode::downgrade(&self_arc)));

                debug_assert!(
                    c.state.read().unwrap().is_some() || !c.parents.read().unwrap().is_empty(),
                    "child needs a parent",
                );

                debug_assert!(
                    r,
                    "\
                        could not remove dropped node as child's parents:\n\
                        \tchild {:p} parent {:p}\
                    ",
                    &*c.inner, &*self_arc,
                );
            }
        }

        if self_arc.registered.load(Ordering::Relaxed) {
            let _r = self_arc
                .registry
                .write()
                .unwrap()
                .remove(&ArcNode::downgrade(&self_arc));
            debug_assert!(_r, "could not remove node");
        } else {
            #[cfg(debug_assertions)]
            {
                assert_eq!(0, self_arc.parents.read().unwrap().len());
                match *self_arc.children.read().unwrap() {
                    Children::NewLeaf => {}
                    _ => panic!("node with children is not registered"),
                }
            }
        }
    }
}

impl<GD, S, P, A, Q, I, M> std::hash::Hash for Node<GD, S, P, A, Q, I, M>
where
    Self: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        h.write_u64(self.hash);
    }
}

impl<GD, S, P, A, Q, I, M> std::cmp::PartialEq for Node<GD, S, P, A, Q, I, M>
where
    Self: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    fn eq(&self, rhs: &Self) -> bool {
        <Self as StateMemory>::eq(self, rhs)
    }
}

impl<GD, S, P, A, Q, I, M> unique_heap::HeapElem for (usize, ArcNode<GD, S, P, A, Q, I, M>)
where
    Node<GD, S, P, A, Q, I, M>: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    // The depth of a node may be modified, so we must store it as a separate field outside the
    // `ArcNode` (required by `UniqueHeap`)
    type Order = usize;
    type UID = (usize, *const Node<GD, S, P, A, Q, I, M>);
    fn order(&self) -> Self::Order {
        self.0
    }
    fn unique_id(&self) -> Self::UID {
        let d = self.0;
        let p = &*self.1.inner as *const _;
        (d, p)
    }
}

impl<GD, S, P, A, Q, I, M> unique_heap::HeapElem for Reverse<(usize, ArcNode<GD, S, P, A, Q, I, M>)>
where
    Node<GD, S, P, A, Q, I, M>: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    // The depth of a node may be modified, so we must store it as a separate field outside the
    // `ArcNode` (required by `UniqueHeap`)
    type Order = Reverse<usize>;
    type UID = (usize, *const Node<GD, S, P, A, Q, I, M>);
    fn order(&self) -> Self::Order {
        Reverse((self.0).0)
    }
    fn unique_id(&self) -> Self::UID {
        let d = (self.0).0;
        let p = &*(self.0).1.inner as *const _;
        (d, p)
    }
}

impl<GD, S, P, A, Q, I, M> Debug for Node<GD, S, P, A, Q, I, M>
where
    Self: StateMemory,
    GD: GameDynamics<Player = P, State = S, Action = A>,
    A: Hash + Eq,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
    Q: Debug,
    S: Debug,
    P: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("ptr", &format!("{:p}", self))
            .field("hash", &self.hash)
            .field("player", &self.player)
            .field("score", self.score.read().unwrap().as_ref().unwrap())
            .field("depth", &self.depth.load(Ordering::Relaxed))
            // .field("state_stored", &*self.state.read().unwrap())
            .field("state", &self.get_state())
            .finish()
    }
}

// Unlike `WeakNode` for which hash and equality comparisons are necessary, `ArcNode` is not
// required as a newtype other for implementation of `Drop` where we rely on `Arc::strong_count`
pub type ArcNode<GD, S, P, A, Q, I, M> = ArcWrap<Node<GD, S, P, A, Q, I, M>>;

#[derive(Debug, Hash, PartialEq)]
pub struct ArcWrap<T: ?Sized + OnDrop> {
    inner: Arc<T>,
}

impl<T: ?Sized + OnDrop> ArcWrap<T> {
    fn downgrade(n: &Self) -> WeakWrap<T> {
        WeakWrap::new(Arc::downgrade(&n.inner))
    }
}

impl<T: ?Sized + OnDrop> Clone for ArcWrap<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: ?Sized + OnDrop> Drop for ArcWrap<T> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            let self_arc = ArcWrap::clone(self);
            <T as OnDrop>::on_drop(self_arc)
        }
    }
}

impl<T: ?Sized + OnDrop> Deref for ArcWrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

// `RefCell` does not implement `Hash`, presumably because once a value is inserted into e.g. a
// map, its hash value cannot change, in the case of a `Node` the values of `hash` and its `state`
// may also not be changed, though a user could do so if `GameDynamics::State` has interior
// mutability; it's the user's responsibility not to mutate the state
pub type WeakNode<GD, S, P, A, Q, I, M> = WeakWrap<Node<GD, S, P, A, Q, I, M>>;

#[derive(Debug)]
pub struct WeakWrap<T: ?Sized> {
    inner: Weak<T>,
}

impl<T: ?Sized + OnDrop> WeakWrap<T> {
    fn new(n: Weak<T>) -> Self {
        Self { inner: n }
    }

    fn upgrade(&self) -> ArcWrap<T> {
        ArcWrap {
            inner: self.inner.upgrade().expect("upgrade failed"),
        }
    }
}

impl<T: ?Sized + OnDrop> std::hash::Hash for WeakWrap<T>
where
    T: Hash,
{
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        Self::upgrade(&self).hash(h);
    }
}

impl<T: ?Sized + OnDrop> std::cmp::PartialEq for WeakWrap<T>
where
    T: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        Self::upgrade(&self) == Self::upgrade(&rhs)
    }
}

impl<T: ?Sized + OnDrop + PartialEq> std::cmp::Eq for WeakWrap<T> {}

impl<T: ?Sized + OnDrop> Deref for WeakWrap<T> {
    type Target = Weak<T>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: ?Sized + OnDrop> Clone for WeakWrap<T> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

#[derive(Debug)]
/// Contains information about a `Tree`'s registry.
pub struct RegistryInfo {
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub len: AtomicUsize,
}

impl RegistryInfo {
    fn new() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }
}

/// An acyclic collection of connected `Node`s with a unique root.
#[derive(Debug)]
pub struct Tree<N: ?Sized + OnDrop, GD: ?Sized> {
    root: RwLock<ArcWrap<N>>,
    // `registry` is a transposition table used to check whether a new node already exists in the
    // tree because there was some other sequence of actions that would lead to the same game
    // state; if another node with the same state is found, then the newly created node is never
    // connected to the tree or entered into the registry
    registry: Arc<RwLock<HashSet<WeakWrap<N>>>>,
    reg_info: RegistryInfo,
    game_dynamics: Arc<GD>,
    prune_lock: RwLock<()>,
}

impl<GD, S, P, A, Q, II, I, M> Tree<Node<GD, S, P, A, Q, I, M>, GD>
where
    Node<GD, S, P, A, Q, I, M>: StateMemory<State = S>,
    GD: GameDynamics<Player = P, State = S, Action = A, Score = Q, ActionIter = II>,
    II: IntoIterator<IntoIter = I, Item = (P, A)>,
    I: Iterator<Item = (P, A)>,
    A: Hash + Eq + Clone,
    S: Hash + PartialEq<S> + Clone,
    P: Hash + PartialEq<P>,
{
    /// Construct a new `Tree`.
    pub fn new(game_dynamics: GD, _: M, first_player: P, root_state: S) -> Self {
        let game_dynamics = Arc::new(game_dynamics);
        let registry = Arc::new(RwLock::new(HashSet::<WeakNode<_, _, _, _, _, _, _>>::new()));
        let root = Node::new_root(
            Arc::clone(&game_dynamics),
            first_player,
            root_state,
            Arc::clone(&registry),
        );
        Tree {
            root: RwLock::new(root),
            registry,
            reg_info: RegistryInfo::new(),
            game_dynamics,
            prune_lock: RwLock::new(()),
        }
    }

    fn step(&self) -> Option<S> {
        let _prune_rlk = self.prune_lock.read().unwrap();
        let node = ArcNode::clone(&*self.root.read().unwrap());
        let state = node.get_state();
        self.step_into(state, node)
    }

    fn step_into(&self, mut node_state: S, mut node: ArcNode<GD, S, P, A, Q, I, M>) -> Option<S> {
        loop {
            let children_rlk = node.children.read().unwrap();
            match *children_rlk {
                Children::NewLeaf => {
                    drop(children_rlk);
                    self.make_branch_wip(&node_state, &node);
                    self.make_branch(&node_state, &node);
                    Node::backprop_scores(&node);
                    return Some(node_state);
                }
                Children::BranchWip(_) => {
                    drop(children_rlk);
                    // `make_branch` returns when `Children::BranchWip` is converted to
                    // `Children::Branch`; so loop again
                    self.make_branch(&node_state, &node);
                }
                Children::Branch(ref map) => {
                    let action =
                        Self::select_node(&self, &node, &node_state, map, SelectNodeState::Explore);

                    // get the selected child node, calculate its state, and keep recursing
                    let next_node = ArcNode::clone(&map.get(&action).unwrap());

                    drop(children_rlk);
                    node = next_node;
                    node_state =
                        GD::apply_action(&*node.game_dynamics, node_state, &action).unwrap();
                }
                Children::None => {
                    // Currently the implementation assumes that the score of a `Terminal` node is
                    // immutable even if the selection path leads there repeatedly
                    // Node::backprop_scores(&node);
                    return None;
                }
            }
        }
    }

    fn select_node(
        &self,
        parent_node: &ArcNode<GD, S, P, A, Q, I, M>,
        parent_node_state: &S,
        children: &HashMap<A, ArcNode<GD, S, P, A, Q, I, M>>,
        purpose: SelectNodeState,
    ) -> A {
        let scores_and_actions = children.iter().map(|(a, child)| {
            // Taking a standard shared reference to the score will not compile because the
            // `Ref<'a,T>` would go out of scope at the end of the closure, and the lifetime of the
            // return value of `<Ref<'a,T> as Deref>::deref` is tied to the lifetime of the
            // `Ref<'a,T>` caller and *not* the lifetime `'a`; this is because `Ref` contains a
            // `BorrowRef` which implements `Drop` to decrement the reference counter; a `Ref`
            // cannot simply give out standard shared references because outstanding references
            // need to be tracked, and therefore the lifetime of the standard shared reference
            // returned by `<Ref<'a,T> as Deref>::deref` is limited to the lifetime of the `Ref`
            // caller; in order to hide the implementation details of the `Ref` which would
            // otherwise be exposed to the user in `GD::select_node`, we map the `Ref` to the score
            // field and pass the mapped `Ref` to the user defined implementation of
            // `GD::select_node`, which accepts any `Deref<Target = Score>`; we hold a read lock to
            // the node's `children` field (outside this function) and lazily acquire a lock on
            // each of the children's `scores` field in `GD::select_node`
            // let s = child.score.read().unwrap().as_ref().unwrap();
            let q = lockref::Ref::new(child.score.read().unwrap(), |q| &**q);
            (q, a)
        });

        GD::select_node(
            &*self.game_dynamics,
            parent_node.score.read().unwrap().as_ref(),
            &parent_node.player,
            parent_node_state,
            purpose,
            scores_and_actions,
        )
    }

    fn create_scored_child(
        &self,
        parent_node: &ArcNode<GD, S, P, A, Q, I, M>,
        player: P,
        action: A,
        state: S,
    ) {
        let node = Node::new_child(parent_node, player, state);

        // check if node is in the registry, if not: add to registry, then calculate score, then
        // connect node to tree
        debug_assert!(node.state.read().unwrap().is_some());
        let mut reg_wlk = self.registry.write().unwrap();
        match reg_wlk.get(&ArcNode::downgrade(&node)) {
            Some(existing_node) => {
                let node = WeakNode::upgrade(existing_node);
                Node::connect_child(parent_node, action, &node);
                drop(reg_wlk);

                self.reg_info.hits.fetch_add(1, Ordering::Relaxed);

                Node::set_min_depth(&node);
                // `node.score` may be `None` but the score will be set before a read lock on
                // `node.score` is available (see `None` arm below)
            }
            None => {
                // acquire a write lock on `node.score` before `reg_wlk` is released so that other
                // threads block on trying to read `node.score` before it is calculated
                let mut score_wlk = node.score.write().unwrap();
                Node::connect_child(parent_node, action, &node);
                Node::register(&node, Some(&mut reg_wlk));
                drop(reg_wlk);

                self.reg_info.misses.fetch_add(1, Ordering::Relaxed);

                // the depth of `parent_node` may have been updated by another thread between the
                // call to `Node::new_child` and connecting the child to the parent.  Since
                // `Node::set_min_depth` only works with children that are connected, the depth of
                // `node` may actually be stale / incorrect, so we update the depth here while we
                // have a write lock on `score`
                Node::set_min_depth(&node);

                // Only run `GD::score_leaf` for nodes that don't exist in the registry
                // it's ok to hold the read lock on `node.state` for an extended period of time (if
                // `GD::score_leaf` is slow) since no write lock is acquired on this field during
                // expansion (a write lock is only acquired on this field during `move_root` /
                // `Drop::drop` and `StateMemory::modify_state`)
                *score_wlk = GD::score_leaf(
                    &*self.game_dynamics,
                    parent_node.score.read().unwrap().as_ref(),
                    &parent_node.player,
                    node.state.read().unwrap().as_ref().unwrap(),
                );
                drop(score_wlk);

                <Node<GD, S, P, A, Q, I, M> as StateMemory>::modify_state(&node.state);
            }
        }
    }

    fn make_branch(&self, parent_state: &S, parent_node: &ArcNode<GD, S, P, A, Q, I, M>) {
        // bracket needed for `debug_assertions` below so there is no deadlock on `children_wlk`
        {
            let mut children_wlk = parent_node.children.write().unwrap();
            // To allow other threads to steal work, we drop `children_wlk` as soon as we no longer
            // need `branch_wip`
            while let Children::BranchWip(ref mut branch_wip) = *children_wlk {
                if let Some((p, a)) = branch_wip.next_unscored() {
                    // a new player / action pair; `GD::apply_action` and
                    // `Self::create_scored_child` could both be slow (depending on user
                    // implementation of `GameDynamics` so we go ahead and drop the `children_wlk`
                    drop(children_wlk);
                    if let Some(state) =
                        GD::apply_action(&*parent_node.game_dynamics, parent_state.clone(), &a)
                    {
                        self.create_scored_child(parent_node, p, a, state);
                        children_wlk = parent_node.children.write().unwrap();
                    } else {
                        // `BranchWip` keeps a counter to ensure all nodes have been created, since
                        // this node won't be included in the `Branch` we need to let `BranchWip`
                        // know
                        children_wlk = parent_node.children.write().unwrap();
                        children_wlk.as_wip_mut().unwrap().decrease_scores_pending();
                    }
                } else if branch_wip.finished() {
                    // no player / action pairs and ready to convert to `Branch`
                    let map = branch_wip.take_scored().unwrap();
                    let notifier = branch_wip.get_notifier();
                    // since `GD::apply_action` is allowed to return `None` we must check whether
                    // the node is actually a terminal node
                    if map.is_empty() {
                        *children_wlk = Children::None;
                    } else {
                        *children_wlk = Children::Branch(map);
                    }
                    drop(children_wlk);
                    notifier.notify_all();
                    break;
                } else {
                    // no more player / action pairs but another thread is still processing a pair
                    let notifier = branch_wip.get_notifier();
                    drop(children_wlk);
                    drop(notifier.wait().unwrap());
                    break;
                }
            }
        }

        #[cfg(debug_assertions)]
        {
            let children_rlk = parent_node.children.read().unwrap();
            match *children_rlk {
                Children::Branch(_) | Children::None => {}
                _ => panic!("unexpected child variant"),
            }
        }
    }

    fn make_branch_wip(&self, parent_state: &S, parent_node: &ArcNode<GD, S, P, A, Q, I, M>) {
        if let ref mut children @ Children::NewLeaf = *parent_node.children.write().unwrap() {
            let players_actions = self
                .game_dynamics
                .available_actions(&parent_node.player, parent_state);

            match players_actions {
                Some(player_acts) => {
                    let branch_wip = BranchWip::new(player_acts.into_iter());
                    *children = Children::BranchWip(branch_wip);
                }
                None => {
                    *children = Children::None;
                }
            }
        }
    }

    fn best_action(&self) -> Status<A> {
        Self::best_action_from(&self, &self.root.read().unwrap())
    }

    fn best_action_from(&self, node: &ArcNode<GD, S, P, A, Q, I, M>) -> Status<A>
    where
        GD: GameDynamics<Score = Q>,
    {
        let children = node.children.read().unwrap();
        Status::from_children(&*children, |_| {
            // as an alternative to panicking for the node.state argument, could use
            // `&node.get_state()`, though this results in an additional `clone` ... and really,
            // `best_action_from` is only called from the root node, which always has a state (i.e.
            // the `expect` is only a problem if this method is not called on the root node)
            Tree::select_node(
                &self,
                &node,
                node.state
                    .read()
                    .unwrap()
                    .as_ref()
                    .expect("this method can only be called on nodes with a state"),
                children.as_map().unwrap(),
                SelectNodeState::Exploit,
            )
        })
    }

    fn apply_action(&self, a: &A) {
        let _prune_wlk = self.prune_lock.write().unwrap();
        let root_new = self.root.read().unwrap().move_root(a);
        *self.root.write().unwrap() = root_new;
    }

    fn apply_best_action(&self) -> Status<A> {
        let best_action = self.best_action();
        if let Status::Action(ref a) | Status::ActionWip(ref a) = best_action {
            self.apply_action(a);
        }
        best_action
    }

    fn get_root_info(&self) -> NodeInfo<S, P, Q>
    where
        <GD as GameDynamics>::Score: Clone,
        <GD as GameDynamics>::Player: Clone,
    {
        self.root.read().unwrap().get_node_info()
    }

    fn get_next_move_info(&self) -> Option<Vec<(A, NodeInfo<S, P, Q>)>>
    where
        Q: Clone,
        P: Clone,
    {
        let info = self
            .root
            .read()
            .unwrap()
            .children
            .read()
            .unwrap()
            .as_map()?
            .iter()
            .map(|(a, c)| (a.clone(), c.get_node_info()))
            .collect();

        Some(info)
    }

    fn find_children_sorted_with_depth(&self) -> Vec<(ArcNode<GD, S, P, A, Q, I, M>, usize)> {
        let node = self.root.read().unwrap();
        let mut sorted = Vec::new();
        let mut visited = HashMap::new();
        Node::find_children_sorted_with_depth(&node, &mut sorted, &mut visited);
        sorted
    }

    fn get_registry_nodes(&self) -> HashSet<WeakNode<GD, S, P, A, Q, I, M>> {
        self.registry.read().unwrap().clone()
    }

    fn get_registry_info(&self) -> &RegistryInfo {
        self.reg_info
            .len
            .store(self.registry.read().unwrap().len(), Ordering::Relaxed);
        &self.reg_info
    }

    fn get_game_dynamics(&self) -> Arc<GD> {
        Arc::clone(&self.game_dynamics)
    }
}

#[cfg(any(test, feature = "test_internals"))]
pub(crate) mod test {
    use super::*;

    #[allow(dead_code)]
    #[test]
    fn test_can_be_dyn() {
        fn test_trait_object_feasible(
            t: &impl SearchTree<GD = impl GameDynamics, Memory = impl StateMemory>,
        ) -> &dyn SearchTree<GD = impl GameDynamics, Memory = impl StateMemory> {
            // check that we can create a `SearchTree` trait object; details on object safety
            // https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
            t
        }
    }

    #[doc(hidden)]
    #[cfg(feature = "test_internals")]
    pub fn get_state<GD, S, P, A, Q, I, M>(n: &Node<GD, S, P, A, Q, I, M>) -> S
    where
        Node<GD, S, P, A, Q, I, M>: StateMemory + OnDrop,
        GD: GameDynamics<Player = P, State = S, Action = A>,
        A: Hash + Eq,
        S: Hash + PartialEq<S> + Clone,
        P: Hash + PartialEq<P>,
    {
        n.get_state()
    }

    // function intended to be used in an external unit test with a constructed `Tree`
    #[doc(hidden)]
    #[cfg(feature = "test_internals")]
    pub fn test_depth_helper<GD, S, P, A, Q, I, M, II>(
        t: &Tree<Node<GD, S, P, A, Q, I, M>, GD>,
    ) -> (
        Vec<(ArcNode<GD, S, P, A, Q, I, M>, usize)>,
        HashMap<*const Node<GD, S, P, A, Q, I, M>, usize>,
    )
    where
        Node<GD, S, P, A, Q, I, M>: StateMemory<State = S>,
        GD: GameDynamics<Player = P, State = S, Action = A, Score = Q, ActionIter = II>,
        II: IntoIterator<IntoIter = I, Item = (P, A)>,
        I: Iterator<Item = (P, A)>,
        A: Clone + Hash + Eq,
        S: Clone + Hash + PartialEq<S>,
        P: Hash + PartialEq<P>,
    {
        let root = t.root.read().unwrap();
        let mut children = Vec::new();
        let mut visited = HashMap::new();
        let depth = Node::find_children_sorted_with_depth(&root, &mut children, &mut visited);
        println!("nnodes: {} depth: {}", children.len(), depth);
        assert_eq!(t.get_registry_nodes().len(), children.len());

        children.iter().for_each(|(c, _)| {
            let d = c.depth.load(Ordering::Relaxed);
            if let Some(children) = c.children.read().unwrap().as_map() {
                children.iter().for_each(|(_, sub_c)| {
                    let sub_d = sub_c.depth.load(Ordering::Relaxed);
                    assert!(d < sub_d);
                });
            }
        });

        eprintln!("Warning: this test can be very slow");
        children.iter().for_each(|(c, d)| {
            assert_eq!(d, visited.get(&c.as_ptr()).unwrap());
            let mut sub_children = Vec::new();
            Node::find_children_sorted_with_depth(&c, &mut sub_children, &mut HashMap::new());
            sub_children
                .iter()
                .rev()
                .skip(1)
                .for_each(|(sub_c, sub_d)| {
                    assert_eq!(sub_d, visited.get(&sub_c.as_ptr()).unwrap());
                    assert!(d > sub_d);
                });
        });
        (children, visited)
    }
}
