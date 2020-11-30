//! A <b>re</b>combining and <b>con</b>current implementation of [Monte Carlo tree
//! search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS).
//!
//! # Features
//!
//! - **_Recombining:_** the tree efficiently handles the fact that multiple different sequences of
//! actions can lead to the same state.  Since nodes can have multiple parents, the underlying data
//! representation is a [directed acyclic
//! graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph).  The evaluation of a new leaf
//! node will be propagated backwards to all parent nodes, rather than simply to nodes on the path
//! along which the leaf was reached.  Nodes that are no longer reachable are pruned from the tree
//! when an action is taken.
//!
//! - **_Concurrent:_** multiple worker threads can grow the tree at the same time.  Threads that
//! are waiting on a new leaf to be created will steal work from the thread responsible for
//! creating the new leaf in order to prevent log jams along the hot path of the Monte Carlo tree.
//! Additional threads become more beneficial with increases in the amount of time required to run
//! [`GameDynamics::score_leaf`] and greater exploration entropy for a given state in
//! [`GameDynamics::select_node`].
//!
//! - **_Topologically aware:_** the sequence of nodes backpropagating their scores is based on the
//! node's position in the tree.  In the process of backpropagating a score from a leaf to the root
//! of the tree, a node does not backpropagate its score to its parents until it has received all
//! updates from its children.  This is desirable because in a recombining tree, some paths from a
//! leaf to a (grand)*parent may pass through more nodes than other paths.  A parent node waits on
//! information to arrive via all possible paths before pushing score updates to its own parents.
//! Nevertheless, backpropagation leads to O(n^2) complexity even with topological awareness (i.e.
//! the nth node could have n-1 parents, all of which must be updated by the MCTS algorithm).  It
//! is possible to increase the performance of deep trees by only backpropagating a score if it
//! represents a meaningful update to the node's prior score.  See
//! [`GameDynamics::backprop_scores`] for further details.
//!
//! - **_Simple and flexible:_** the interface has a high degree of flexibility and is easy to
//! implement for various discrete move games.  Games can have one or many players; different
//! players can have different evaluation functions (e.g. a player could store its score as a
//! string or bytes array while another player simply uses a float).
//!
//! - **_Configurable:_** Memory usage for game states that are memory intensive can be reduced by
//! choosing to recompute them or to store them as a hash only (at the cost of a possible hash
//! collision that results in a suboptimal decision). See [`state_memory`].
//!
//! - **_Safe and self-contained:_** the library is written entirely in safe Rust without external
//! dependencies (other than the Rust Standard Library).
//!
//! # Usage
//!
//! Using this library mainly requires implementing the [`GameDynamics`] trait for a game.
//! Interacting with the [`Tree`] is accomplished via the [`SearchTree`] trait.
//!
//! Use of the library via a trait object is made possible via [`DynGD`] since `GameDynamics` is
//! automatically implemented for types that *dereference* to types that implement `DynGD`.
//!
//! See the provided [implementation][nim_url] of Nim and the high-level code below for a
//! demonstration of the library interface.
//!
//! ```no_run
//! use recon_mcts::prelude::*;
//! # use std::ops::Deref;
//!
//! struct MyGame;
//!
//! // `Clone` is generally optional, but required for some methods such as
//! // `SearchTree::get_root_info`
//! #[derive(Clone, Debug, Hash, PartialEq)]
//! enum Player {
//!     P1,
//!     P2,
//! }
//!
//! impl GameDynamics for MyGame {
//!     type Player = Player;
//!     type State = usize;
//!     type Action = usize;
//!     type Score = f64;
//!     type ActionIter = Vec<(Self::Player, Self::Action)>;
//!
//!     fn available_actions(
//!         &self,
//!         player: &Self::Player,
//!         state: &Self::State,
//!     ) -> Option<Self::ActionIter> {
//!         todo!()
//!     }
//!
//!     fn apply_action(
//!         &self,
//!         state: Self::State,
//!         action: &Self::Action,
//!     ) -> Option<Self::State> {
//!         todo!()
//!     }
//!
//!     fn select_node<II, Q, A>(
//!         &self,
//!         parent_score: Option<&Self::Score>,
//!         parent_player: &Self::Player,
//!         parent_node_state: &Self::State,
//!         purpose: SelectNodeState,
//!         scores_and_actions: II,
//!     ) -> Self::Action
//!     where
//!         II: IntoIterator<Item = (Q, A)>,
//!         Q: Deref<Target = Option<Self::Score>>,
//!         A: Deref<Target = Self::Action>,
//!     {
//!         todo!()
//!     }
//!
//!     fn backprop_scores<II, Q>(
//!         &self,
//!         player: &Self::Player,
//!         score_current: Option<&Self::Score>,
//!         child_scores: II,
//!     ) -> Option<Self::Score>
//!     where
//!         II: IntoIterator<Item = Q>,
//!         Q: Deref<Target = Self::Score>,
//!     {
//!         todo!()
//!     }
//!
//!     fn score_leaf(
//!         &self,
//!         parent_score: Option<&Self::Score>,
//!         parent_player: &Self::Player,
//!         state: &Self::State,
//!     ) -> Option<Self::Score> {
//!         todo!()
//!     }
//! }
//!
//! let tree = Tree::new(MyGame, GetState, Player::P1, 0);
//!
//! loop {
//!     for _ in 0..9 {
//!         tree.step();
//!     }
//!
//!     match tree.apply_best_action() {
//!         Status::Terminal => break,
//!         _ => { dbg!(tree.get_root_info()); }
//!     }
//! }
//! ```
//!
//! # Contribute
//!
//! You can browse the repo and make contributions on [GitHub][repo_url].
//!
//! [![repo_badge]][repo_url]
//! [![mit_badge]][mit_url]
//!
//! [repo_badge]: https://img.shields.io/badge/repo-github-blue.svg
//! [repo_url]: https://github.com/trtsl/recon_mcts
//! [mit_badge]: https://img.shields.io/badge/license-MIT-blue.svg
//! [mit_url]: https://github.com/trtsl/recon_mcts/blob/master/LICENSE
//! [nim_url]: ../recon_mcts_test_nim/index.html

// Helpful refernces:
// https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/
// https://tor-lattimore.com/downloads/book/book.pdf
// http://incompleteideas.net/book/RLbook2018.pdf

#![forbid(unsafe_code)]
#![warn(
    rust_2018_idioms,
    missing_debug_implementations,
    missing_docs,
    broken_intra_doc_links
)]

mod game_dynamics;
mod lockref;
mod ref_iter;
mod tree;
mod unique_heap;

#[cfg(feature = "two_player")]
mod map_maybe;

#[doc(inline)]
pub use prelude::*;

#[doc(hidden)]
pub mod prelude {
    pub use crate::game_dynamics::{BaseGD, DynGD, GameDynamics, SelectNodeState};
    pub use crate::tree::state_memory::{self, GetState, HashOnly, StateMemory, StoreState};
    pub use crate::tree::{
        ArcNode, ArcWrap, Node, NodeInfo, OnDrop, RegistryInfo, SearchTree, Status, Tree, WeakNode,
        WeakWrap,
    };

    pub use crate::tree::NodeAlias;
    pub use crate::tree::TreeAlias;

    #[cfg(feature = "test_internals")]
    pub use crate::tree::test::*;
}
