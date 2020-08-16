use std::cell::{Ref, RefCell};
use std::ops::Deref;

use crate::ref_iter::RefIterator;

/// A flag indicating whether an action is being evaluated for exploration or exploitation.
pub enum SelectNodeState {
    Explore,
    Exploit,
}

// It's currently not possible to turn `GameDynamics` into a trait object because `select_node` and
// `backprop_scores` use generic parameters.  In particular, the underlying generic type has an
// iterator over `Node<GD, State, Player, Action, Score, I, M>` where `M` is a true generic
// parameter (i.e. not specified as an associate type).  To make `GameDynamics` object safe, a new
// trait parameter or associated type is needed.  Alternatively, there are proposals such as:
// https://internals.rust-lang.org/t/pre-rfc-expand-object-safety/12693
/// Requires implementation by the user in order to provide the rules of the game and desired
/// specifics of the algorithm.
///
/// # Implementation Note
///
/// The associated types should generally not be types that allow interior mutability as mutating
/// them in the methods provided by the `GameDynamics` trait may result in logic error.  However,
/// in some implementations it may be desirable to use a
/// [`GameDynamics::Score`](#associatedtype.Score) type with interior mutability, e.g. in order to
/// modify the number of times it was selected in
/// [`GameDynamics::select_node`](#tymethod.select_node).
pub trait GameDynamics {
    /// Most likely an enum to designate which player is allowed to select an action at the current
    /// node.  For a one player game, this type can simply be the unit `()`.
    type Player;
    /// A type that complete describes the state of the game.  The implementation of the
    /// `GameDynamics` must ensure that graph formed by the search tree is acyclic.  If the game
    /// (e.g. chess) allows the same board position to be reached multiple times, some value within
    /// the state could track how many times the state has been reached.  This is often required
    /// anyway because repeat board position are actually different game states (e.g. in chess,
    /// reaching the same board position three times results in a tie, so the game state is
    /// actually distinct despite the position of pieces on the board being identical).
    type State;
    /// Most likely an enum or integer value to describe the action that transitions from one state
    /// to another.
    type Action;
    /// A type that represents an evaluation of `Self::State`.  This would commonly be a float, but
    /// more elaborate types may be useful for some games, e.g. those with more than two players.
    type Score;
    /// An iterator used to establish the available actions and next player (note that the player
    /// can be different for different actions).
    ///
    /// If the returned iterator contains a closure, it will need to be boxed on stable Rust until
    /// `#![feature(type_alias_impl_trait)]` is stabilized (see
    /// [#63063](https://github.com/rust-lang/rust/issues/63063) and
    /// [#63066](https://github.com/rust-lang/rust/issues/63066)).
    type ActionIter: IntoIterator<Item = (Self::Player, Self::Action)>;

    /// Convert an input state into the available actions / moves. Return `None` if the game is
    /// over.  If the game is over, the [`GameDynamics::score_leaf`] method
    /// will be called to evaluate the state
    ///
    /// [`GameDynamics::score_leaf`]: #tymethod.score_leaf
    fn available_actions(
        &self,
        player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter>;

    /// Modify the state input (i.e. game board) with an action.
    ///
    /// In some games it may be easier to determine whether an action is valid by trying to apply
    /// it and then assess whether it succeeds or fails, we allow
    /// [`GameDynamics::apply_action`] to return on `Option::None` in case
    /// of failure (in such a case the action will be treated as though it was never included in
    /// the output of `available_actions`).
    ///
    /// Note that the user must ensure that for any `State` the same state can never be reached
    /// again by applying any number of `Action`s obtained via `available_actions` (i.e. the user
    /// must ensure the tree graph is acyclic)
    ///
    /// [`GameDynamics::apply_action`]: #tymethod.apply_action
    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State>;

    /// Select the action to take based on the scores.
    ///
    /// Note that because `scores_and_actions` are received in non-deterministic order, the result
    /// of the MCTS algorithm may also be non-deterministic if there is a tie with respect to the
    /// selected score for two different actions.
    ///
    /// The `SelectNodeState` maybe be used to add an additional exploration parameter when
    /// searching for the best action.
    ///
    /// `parent_score` is included because MCTS commonly uses the parent's count of number of
    /// visits in the selection phase.
    ///
    /// `parent_node_state` is not typically a required argument, but can be helpful if performing
    /// only a partial expansion of nodes is desired upon reaching a leaf node and recalculating
    /// the state is preferred over storing it in an enum. See additional info in
    /// [`GameDynamics::score_leaf`].
    ///
    /// [`GameDynamics::score_leaf`]: #tymethod.score_leaf
    // Instead of using a `Deref<Target = T>` we could probably just lock the interface into
    // providing `RwLockReadGuard<T>` so we wouldn't need a separate `DynGD` trait for dynamic
    // dispatch ... but that would be less fun
    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        parent_node_state: &Self::State,
        purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>;

    /// Score a parent node based on its child nodes.
    ///
    /// `score_current` is passed to enable comparison with the calculated score so that if the
    /// difference is insignificant, the backpropagation to the root node can be cut off by
    /// returning a `None`.  For deep trees this may provide a decent performance boost.  The root
    /// note does not have a score initially, so an `Option<&Self::Score>` is used for
    /// `score_current`.
    ///
    /// Returning a `None` does not imply that `score_current` will be `None` on the next iteration
    /// -- that would only be the case *iff* it was a `None` on the current iteration and a `None` is
    /// returned.
    // This function is the reason `Self::Player` is a separate field from `Self::State`.
    // `Self::State` may be large such that it is only stored for the root node; therefore,
    // `Self::State` is not available to `backprop_scores`.  However, `Self::Player` could be
    // useful for giving different player different backpropagation methodologies.
    fn backprop_scores<II, Q>(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = Q>,
        Q: Deref<Target = Self::Score>;

    /// Take a leaf node's state and assign the node a score, whether via simulation or otherwise.
    ///
    /// Note that for a leaf where no actions are possible (i.e. a terminal node) the score should
    /// be considered immutable.  Changes to the score of such a terminal node on subsequent
    /// simulation runs will *not* result in the updates being backpropagated through the tree.
    ///
    /// # Implementation Note:
    ///
    /// The library currently expands all branches of a leaf, though this can effectively be
    /// modified to a delayed scoring / rollout by using a type such as the following for the
    /// associated type `Score` and then evaluating the `Score` in [`GameDynamics::select_node`] if
    /// a `MyScore::State` is encountered.
    /// ```ignore
    /// // pseudo-code
    /// impl GameDynamics for MyGame {
    ///     // `MyRealScore` will need to be updated via a shared ref (i.e. it must be either an
    ///     // atomic variable or wrapped in a type that provides interior mutability). This is an
    ///     // exception to the rule that the associated types should not be mutated by the trait
    ///     // methods for cases where a delayed rollout is desired.
    ///     type Score = enum MyScore {
    ///         State(Self::State),
    ///         Score(MyRealScore),
    ///     };
    ///     ...
    /// }
    /// ```
    /// Alternatively, one could return a `None` and derive the state in
    /// [`GameDynamics::select_node`] using [`GameDynamics::available_actions`] if a `None` is
    /// encountered.  The state will be calculated for the leaf node irrespective of whether the
    /// scoring is performed by [`GameDynamics::score_leaf`] because it is used in checking whether
    /// the same state already exists in the tree (i.e. encountered via a different sequence of
    /// actions).
    ///
    /// [`GameDynamics::available_actions`]: #tymethod.available_actions
    /// [`GameDynamics::select_node`]: #tymethod.select_node
    /// [`GameDynamics::score_leaf`]: #tymethod.score_leaf
    fn score_leaf(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score>;
}

/// A trait that can be used to implemented [`DynGD`] without implementing [`GameDynamics`].
/// `BaseGD` is automatically implemented for types that implemented `GameDynamics`.
///
/// `DynGD` is a supertrait of `BaseGD`.
///
/// For descriptions of the associated types and required methods see [`GameDynamics`].
///
/// [`DynGd`]: trait.DynGD.html
/// [`GameDynamics`]: trait.GameDynamics.html
pub trait BaseGD {
    type Player;
    type State;
    type Action;
    type Score;
    type ActionIter: IntoIterator<Item = (Self::Player, Self::Action)>;

    fn available_actions(
        &self,
        player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter>;

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State>;

    fn score_leaf(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score>;
}

impl<T> BaseGD for T
where
    T: GameDynamics,
{
    type Player = <T as GameDynamics>::Player;
    type State = <T as GameDynamics>::State;
    type Action = <T as GameDynamics>::Action;
    type Score = <T as GameDynamics>::Score;
    type ActionIter = <T as GameDynamics>::ActionIter;

    #[inline(always)]
    fn available_actions(
        &self,
        player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter> {
        <T as GameDynamics>::available_actions(&self, player, state)
    }

    #[inline(always)]
    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        <T as GameDynamics>::apply_action(&self, state, action)
    }

    #[inline(always)]
    fn score_leaf(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        <T as GameDynamics>::score_leaf(&self, parent_score, parent_player, state)
    }
}

/// A supertrait of [`BaseGD`].  Its purpose is to implement `GameDynamics` for trait objects.
/// Implementing `DynGD` is only required if dynamic dispatch is needed.
///
/// Since [`GameDynamics::select_node`] and [`GameDynamics::backprop_scores`] have generic
/// parameters, a type implementing [`GameDynamics`] cannot be used as a trait object directly.
/// However, `GameDynamics` is automatically implemented for types that *dereference* to a type
/// implementing `DynGD` (see [`GameDynamics` Implementors]).
///
/// `BaseGD` is automatically implemented for types that implement `GameDynamics`, but can also be
/// implemented directly without implementing `GameDynamics` if static dispatch is not required
/// (i.e. implementing `BaseGD` and `DynGD` without implementing `GameDynamics` will allow for
/// dynamic dispatch but not static dispatch).  For descriptions of the associated types and
/// required methods see [`GameDynamics`].
///
/// See the provided [`implementation`](../recon_mcts_test_nim/index.html) of Nim for a demonstration
/// of using static and dynamic dispatch.
///
/// # Implementation Note
///
/// Using `DynGD` for dynamic dispatch does not result in additional heap allocations.  This is
/// accomplished via the use of a single `RefCell` on the stack which is referenced by all
/// `Ref<T>`s returned by an `IntoIterator` type in the `DynGD` trait methods.  As a result,
/// calling `Deref::deref` on a `Ref` that is not the `Ref` most recently returned by the
/// iterator will result in a panic.  If out-of-sequence access is required, the implementation must
/// store the dereferenced values rather than the `Ref`s.
///
/// [`BaseGD`]: trait.BaseGD.html
/// [`GameDynamics`]: trait.GameDynamics.html
/// [`GameDynamics::select_node`]: trait.GameDynamics.html#tymethod.select_node
/// [`GameDynamics::backprop_scores`]: trait.GameDynamics.html#tymethod.backprop_scores
/// [`GameDynamics` Implementors]: trait.GameDynamics.html#implementors
#[allow(clippy::type_complexity)]
pub trait DynGD: BaseGD {
    fn select_node(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        parent_node_state: &Self::State,
        purpose: SelectNodeState,
        scores_and_actions: &mut dyn Iterator<
            Item = (Ref<'_, Option<Self::Score>>, Ref<'_, Self::Action>),
        >,
    ) -> Self::Action;

    fn backprop_scores(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores: &mut (dyn Iterator<Item = Ref<'_, Self::Score>>),
    ) -> Option<Self::Score>;
}

impl<R, T> GameDynamics for R
where
    R: ?Sized + Deref<Target = T>,
    T: ?Sized + DynGD,
{
    type Player = <T as BaseGD>::Player;
    type State = <T as BaseGD>::State;
    type Action = <T as BaseGD>::Action;
    type Score = <T as BaseGD>::Score;
    type ActionIter = <T as BaseGD>::ActionIter;

    #[inline(always)]
    fn available_actions(&self, player: &T::Player, state: &T::State) -> Option<T::ActionIter> {
        <T as BaseGD>::available_actions(self, player, state)
    }

    #[inline(always)]
    fn apply_action(&self, state: T::State, action: &T::Action) -> Option<T::State> {
        <T as BaseGD>::apply_action(self, state, action)
    }

    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&T::Score>,
        parent_player: &T::Player,
        parent_node_state: &T::State,
        purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> T::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<T::Score>>,
        A: Deref<Target = T::Action>,
    {
        // using `ref_iter` allows creating a trait object save value from the generic
        // `Deref<Target = ...>` without allocating on the heap
        let reserved_space = RefCell::new(None);
        let scores = scores_and_actions
            .clone()
            .into_iter()
            .map(|qa| qa.0)
            .ref_iter(&reserved_space)
            .map(|q| Ref::map(q, Deref::deref));

        let reserved_space = RefCell::new(None);
        let actions = scores_and_actions
            .into_iter()
            .map(|qa| qa.1)
            .ref_iter(&reserved_space)
            .map(|a| Ref::map(a, Deref::deref));

        let mut qa = scores.zip(actions);

        <T as DynGD>::select_node(
            self,
            parent_score,
            parent_player,
            parent_node_state,
            purpose,
            &mut qa,
        )
    }

    fn backprop_scores<II, Q>(
        &self,
        player: &T::Player,
        score_current: Option<&T::Score>,
        child_scores: II,
    ) -> Option<T::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = Q>,
        Q: Deref<Target = T::Score>,
    {
        let reserved_space = RefCell::new(None);
        let mut child_scores = child_scores
            .into_iter()
            .ref_iter(&reserved_space)
            .map(|q| Ref::map(q, Deref::deref));

        <T as DynGD>::backprop_scores(self, player, score_current, &mut child_scores)
    }

    #[inline(always)]
    fn score_leaf(
        &self,
        parent_score: Option<&T::Score>,
        parent_player: &T::Player,
        state: &T::State,
    ) -> Option<T::Score> {
        <T as BaseGD>::score_leaf(self, parent_score, parent_player, state)
    }
}

// An interface for two player games that wraps the generic `GameDynamics` interface above; Not yet
// ready for prime time (i.e. untested and possibly stale)
#[cfg(feature = "two_player")]
mod untested {
    use super::*;
    use crate::map_maybe;

    pub enum PlayerId {
        Player1,
        Player2,
    }

    pub trait GameDynamics2P {
        type Player;
        type State;
        type Action;
        type ScoreP1: 'static;
        type ScoreP2: 'static;
        type ActionIter: IntoIterator<Item = (Self::Player, Self::Action)>;

        fn player_id(player: &Self::Player) -> PlayerId;

        fn available_actions(
            &self,
            player: &Self::Player,
            state: &Self::State,
        ) -> Option<Self::ActionIter>;

        fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State>;

        fn select_node_player1<'a, II, Q, A>(
            purpose: SelectNodeState,
            scores_and_actions: II,
        ) -> Option<Self::Action>
        where
            II: Clone + IntoIterator<Item = (Q, A)>,
            Q: map_maybe::MapMaybe<'a, Target = Self::ScoreP1>,
            A: Deref<Target = Self::Action>;

        fn select_node_player2<'a, II, Q, A>(
            purpose: SelectNodeState,
            scores_and_actions: II,
        ) -> Option<Self::Action>
        where
            II: Clone + IntoIterator<Item = (Q, A)>,
            Q: map_maybe::MapMaybe<'a, Target = Self::ScoreP2>,
            A: Deref<Target = Self::Action>;

        fn backprop_scores<II, Q>(
            &self,
            player: &Self::Player,
            child_scores: II,
        ) -> Option<(Self::ScoreP1, Self::ScoreP2)>
        where
            II: Clone + IntoIterator<Item = Q>,
            Q: Deref<Target = (Self::ScoreP1, Self::ScoreP2)>;

        fn score_leaf_player1(&self, state: &Self::State) -> Self::ScoreP1;
        fn score_leaf_player2(&self, state: &Self::State) -> Self::ScoreP2;
    }

    impl<T> GameDynamics for T
    where
        Self: GameDynamics2P,
        // <Self as GameDynamics2P>::ScoreP1: 'static,
        // <Self as GameDynamics2P>::ScoreP2: 'static,
    {
        type Player = <Self as GameDynamics2P>::Player;
        type State = <Self as GameDynamics2P>::State;
        type Action = <Self as GameDynamics2P>::Action;
        type Score = (
            <Self as GameDynamics2P>::ScoreP1,
            <Self as GameDynamics2P>::ScoreP2,
        );
        type ActionIter = <Self as GameDynamics2P>::ActionIter;

        fn available_actions(
            &self,
            player: &Self::Player,
            state: &Self::State,
        ) -> Option<Self::ActionIter> {
            <Self as GameDynamics2P>::available_actions(self, player, state)
        }

        fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
            <Self as GameDynamics2P>::apply_action(self, state, action)
        }

        fn select_node<II, Q, A>(
            &self,
            _parent_score: Option<&Self::Score>,
            parent_player: &Self::Player,
            _parent_node_state: &Self::State,
            purpose: SelectNodeState,
            scores_and_actions: II,
        ) -> Option<Self::Action>
        where
            II: Clone + IntoIterator<Item = (Q, A)>,
            Q: Deref<Target = Option<Self::Score>>,
            A: Deref<Target = Self::Action>,
        {
            match <Self as GameDynamics2P>::player_id(parent_player) {
                PlayerId::Player1 => <Self as GameDynamics2P>::select_node_player1(
                    purpose,
                    scores_and_actions.into_iter().map(|(q, a)| {
                        let q = map_maybe::Ref::new(
                            q,
                            (|x| &x.0) as fn(&Self::Score) -> &<Self as GameDynamics2P>::ScoreP1,
                        );
                        (q, a)
                    }),
                ),
                PlayerId::Player2 => <Self as GameDynamics2P>::select_node_player2(
                    purpose,
                    scores_and_actions.into_iter().map(|(q, a)| {
                        let q = map_maybe::Ref::new(
                            q,
                            (|x| &x.1) as fn(&Self::Score) -> &<Self as GameDynamics2P>::ScoreP2,
                        );
                        (q, a)
                    }),
                ),
            }
        }

        fn backprop_scores<II, Q>(
            &self,
            player: &Self::Player,
            _score_current: Option<&Self::Score>,
            child_scores: II,
        ) -> Option<Self::Score>
        where
            II: Clone + IntoIterator<Item = Q>,
            Q: Deref<Target = Self::Score>,
        {
            <Self as GameDynamics2P>::backprop_scores(self, player, child_scores)
        }

        fn score_leaf(
            &self,
            _parent_score: Option<&Self::Score>,
            _parent_player: &Self::Player,
            state: &Self::State,
        ) -> Option<Self::Score> {
            Some((
                <Self as GameDynamics2P>::score_leaf_player1(self, state),
                <Self as GameDynamics2P>::score_leaf_player2(self, state),
            ))
        }
    }
}
