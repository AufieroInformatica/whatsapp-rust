//! What the VoIP runtime parks on the core, and the hooks it fills.
//!
//! Every one of these used to be a `Client` field: five of them, each with its
//! own `cfg`, plus the branches that built and tore them down. They are now one
//! `Subsystem` impl (`src/client/subsystem.rs`), which is why nothing outside
//! `src/voip/`, `src/client/voip.rs` and `src/handlers/call.rs` names VoIP any
//! more.

use std::sync::Arc;

use async_lock::Mutex;
use wacore::stats::CollectionStats;
use wacore_binary::NodeRef;

use crate::client::Client;
use crate::client::subsystem::Subsystem;

/// What this subsystem reports to `Client::memory_report`, named once so a
/// lookup and the hook below cannot drift apart, and so a caller spells a
/// constant instead of a string.
pub mod collections {
    use super::Voip;
    use crate::client::SubsystemCollection;
    use crate::client::subsystem::Subsystem;

    /// Admission snapshots retained while a call-link join ACK is in flight.
    pub const PENDING_LINK_UPDATES: SubsystemCollection =
        SubsystemCollection::new(Voip::NAME, "pending_link_updates");
    /// Active and ringing calls, with their bounded pre-offer group controls.
    pub const ACTIVE_CALLS: SubsystemCollection =
        SubsystemCollection::new(Voip::NAME, "active_calls");
    /// Outgoing calls parked until the server sends the relay that owns them.
    pub const PENDING_OUTGOING: SubsystemCollection =
        SubsystemCollection::new(Voip::NAME, "pending_outgoing");
}

/// VoIP's attachment to the core. It claims no notification type: calls arrive
/// as `<call>` stanzas, which the stanza router already dispatches to
/// `CallHandler`, so the two hooks below are all it takes from the core.
pub(crate) struct Voip;

impl Subsystem for Voip {
    type State = VoipState;

    const NAME: &'static str = "voip";

    fn on_connection_ending(client: &Client, will_reconnect: bool) {
        client
            .voip_state()
            .direct
            .signaling_expected_back
            .store(will_reconnect, std::sync::atomic::Ordering::Release);
    }

    /// Signaling is connection-scoped, so a call that has not reached live media ends with its
    /// connection. A call whose relay session is up does not need the socket to carry audio, so
    /// when the supervision loop is about to reconnect it is suspended instead and re-attached by
    /// [`on_connected`](Self::on_connected); everything that is not a plain reconnect (logout,
    /// replaced stream, ban, shutdown) still ends every call here and now.
    async fn on_connection_cleanup(client: &Client) {
        let expected_back = client
            .voip_state()
            .direct
            .signaling_expected_back
            .swap(false, std::sync::atomic::Ordering::AcqRel)
            && !client.shutdown_signal().is_fired();
        let host = expected_back
            .then(|| client.self_weak.get().and_then(std::sync::Weak::upgrade))
            .flatten();
        match host {
            Some(host) => {
                let host: Arc<dyn super::direct::DirectCallHost> = host;
                super::direct::suspend_signaling(&host, super::direct::SIGNALING_RESUME_TIMEOUT);
            }
            None => super::direct::disconnect(client),
        }
    }

    fn on_connected(client: &Client) {
        super::direct::resume_signaling(client);
    }

    fn on_session_ended(client: &Client) {
        super::direct::disconnect(client);
    }

    fn on_response(client: &Client, node: &NodeRef<'_>) {
        client.bind_pending_call_link_join_ack(node);
    }

    fn memory(
        state: &VoipState,
        now: impl FnOnce() -> wacore::time::Instant,
    ) -> Vec<(&'static str, CollectionStats)> {
        let pending_link_updates = state
            .pending_call_link_joins
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .memory_stats();
        let pending_outgoing = state
            .pending_outgoing_calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len() as u64;
        vec![
            (
                collections::PENDING_LINK_UPDATES.collection,
                pending_link_updates,
            ),
            (
                collections::ACTIVE_CALLS.collection,
                state.direct.call_registry.memory_stats_with_now(now),
            ),
            // Entry count only: each parked call retains engine material this
            // side cannot size without walking it.
            (
                collections::PENDING_OUTGOING.collection,
                CollectionStats::new(pending_outgoing, 0),
            ),
        ]
    }
}

/// Everything one client retains for VoIP.
pub(crate) struct VoipState {
    pub(crate) direct: super::direct::DirectCallState,
    /// Admission snapshots that can race a call-link join ACK before its call id is registered.
    /// Kept client-side so `wacore` never has to authorize a call it does not know.
    pub(crate) pending_call_link_joins:
        Arc<std::sync::Mutex<crate::client::voip::PendingCallLinkJoins>>,
    /// Serializes call-link joins until the ACK reveals which call id owns any admission state
    /// buffered during the request, so a bounded overflow stays tied to one join. Not shared: the
    /// one caller locks it through `&Client`, unlike the striped answer lanes, which hand out an
    /// owned guard via `lock_arc` and so have to be `Arc`.
    pub(crate) pending_call_link_join_lane: Mutex<()>,
    /// How a relay endpoint becomes a media transport, when the platform supplies its own.
    ///
    /// `None` means "use whatever this build has": the UDP/DTLS/SCTP dialer on a `voip-relay-native`
    /// build, and nothing at all anywhere else -- a page that has not installed one gets a call
    /// that fails at setup with a reason, rather than a crate that would not compile.
    ///
    /// A `std` lock over one `Option`, taken for the length of a clone. It is written once during
    /// assembly in every use anyone has, but making it a constructor argument would put a VoIP type
    /// in `ClientBuilder` for every consumer that never places a call.
    pub(crate) relay_transport_provider:
        std::sync::Mutex<Option<Arc<dyn wacore::voip_control::transport::RelayTransportProvider>>>,
    /// What the native dialer opens its datagram flow through, when not a direct UDP socket.
    /// Consulted only when no provider is installed, since a provider replaces the native dialer
    /// outright.
    #[cfg(feature = "voip-relay-native")]
    pub(crate) relay_datagram_dialer:
        std::sync::Mutex<Option<Arc<dyn crate::voip::transport::DatagramDialer>>>,
}

impl Default for VoipState {
    fn default() -> Self {
        Self {
            direct: super::direct::DirectCallState::new(Arc::new(
                wacore::voip_control::registry::CallRegistry::new(),
            )),
            pending_call_link_joins: Arc::new(std::sync::Mutex::new(Default::default())),
            pending_call_link_join_lane: Mutex::new(()),
            relay_transport_provider: std::sync::Mutex::new(None),
            #[cfg(feature = "voip-relay-native")]
            relay_datagram_dialer: std::sync::Mutex::new(None),
        }
    }
}

impl std::ops::Deref for VoipState {
    type Target = super::direct::DirectCallState;

    fn deref(&self) -> &Self::Target {
        &self.direct
    }
}

impl Client {
    /// The VoIP state parked during client assembly. Named `voip_state` because
    /// `Client::voip()` is already the public facade accessor.
    pub(crate) fn voip_state(&self) -> &VoipState {
        self.subsystem::<Voip>()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::Arc;

    use wacore::voip_control::{CallPhase, CallSession};
    use wacore_binary::{Jid, Server};

    use crate::client::{Client, subsystem};

    /// A call whose media is live, registered straight into the client's call registry.
    async fn client_with_live_call() -> Arc<Client> {
        let client = crate::test_utils::create_test_client().await;
        let registry = client.voip_state().direct.registry().clone();
        let generation = registry.insert(CallSession::new_outgoing(
            "LIVE",
            Jid::new("222222222222222", Server::Lid),
            Jid::new("111111111111111", Server::Lid),
        ));
        assert!(registry.transition_if_current("LIVE", generation, CallPhase::Calling));
        assert!(registry.direct_media_connected_if_current("LIVE", generation));
        client
    }

    fn call_alive(client: &Client) -> bool {
        client
            .voip_state()
            .direct
            .registry()
            .generation_of("LIVE")
            .is_some()
    }

    #[tokio::test]
    async fn a_connection_that_will_be_replaced_leaves_a_live_call_suspended() {
        let client = client_with_live_call().await;

        subsystem::on_connection_ending(&client, true);
        client.cleanup_connection_state().await;

        assert!(call_alive(&client));
        assert!(
            client
                .voip_state()
                .direct
                .registry()
                .is_signaling_suspended("LIVE")
        );

        subsystem::on_connected(&client);
        assert!(call_alive(&client));
        assert!(
            !client
                .voip_state()
                .direct
                .registry()
                .is_signaling_suspended("LIVE")
        );
    }

    #[tokio::test]
    async fn a_connection_that_will_not_be_replaced_ends_the_call_at_once() {
        let client = client_with_live_call().await;

        subsystem::on_connection_ending(&client, false);
        client.cleanup_connection_state().await;

        assert!(!call_alive(&client), "a logout, a ban or a conflict");
    }

    #[tokio::test]
    async fn cleanup_without_a_verdict_from_the_run_loop_ends_the_call() {
        let client = client_with_live_call().await;

        client.cleanup_connection_state().await;

        assert!(
            !call_alive(&client),
            "only a connection the run loop says will be replaced is waited for"
        );
    }

    #[tokio::test]
    async fn the_expectation_does_not_outlive_the_cleanup_it_was_for() {
        let client = client_with_live_call().await;
        subsystem::on_connection_ending(&client, true);
        subsystem::on_connection_ending(&client, false);

        client.cleanup_connection_state().await;

        assert!(!call_alive(&client));
    }

    #[tokio::test]
    async fn a_shutdown_ends_the_call_even_when_a_reconnect_was_expected() {
        let client = client_with_live_call().await;
        subsystem::on_connection_ending(&client, true);
        client.signal_shutdown_sync();

        client.cleanup_connection_state().await;

        assert!(!call_alive(&client));
    }

    #[tokio::test]
    async fn the_end_of_the_run_loop_ends_a_suspended_call() {
        let client = client_with_live_call().await;
        subsystem::on_connection_ending(&client, true);
        client.cleanup_connection_state().await;
        assert!(call_alive(&client));

        subsystem::on_session_ended(&client);

        assert!(!call_alive(&client), "no connection is coming back");
    }
}
