//! Host capabilities for direct-call control. Signal and connection ownership remain with the host.

use std::sync::Arc;

use async_lock::Mutex as AsyncMutex;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use wacore::message_processing::EncType;
use wacore::runtime::{BoxFuture, Runtime};
use wacore::stanza::call::OfferDeviceKey;
use wacore::sync_marker::MaybeSendSync;
use wacore::voip_control::registry::CallRegistry;
use wacore_binary::{Jid, Node};

use crate::client::CallError;

const ANSWER_TRANSITION_LANES: usize = 16;

/// How long a call with live media waits for signaling to come back after the connection drops.
///
/// A reconnect is a backoff, a connect through whatever proxy the host uses (bounded at 20 s) and a
/// login, so one attempt that succeeds takes a few seconds and one that fails and is retried can
/// take most of a minute. 30 s covers a single failed attempt; beyond it the connection is more
/// likely gone for good than slow, and a call whose peer may have hung up unseen is better ended
/// than left running.
pub const SIGNALING_RESUME_TIMEOUT: Duration = Duration::from_secs(30);

/// Encryption results retain the full rung device set even when some devices cannot be encrypted.
pub struct DirectCallKeys {
    pub device_keys: Vec<OfferDeviceKey>,
    pub ring_devices: Vec<Jid>,
    pub device_identity: Option<Vec<u8>>,
}

/// The host owns its live ratchets, key-store transactions, connection and response correlation.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait DirectCallHost: MaybeSendSync + 'static {
    fn call_state(&self) -> &DirectCallState;
    fn call_registry(&self) -> Arc<CallRegistry> {
        self.call_state().call_registry.clone()
    }
    async fn lock_answer_transition(&self, id: &str) -> async_lock::MutexGuardArc<()> {
        self.call_state().transition(id).lock_arc().await
    }
    fn runtime(&self) -> Arc<dyn Runtime>;
    fn own_lid(&self) -> Option<Jid>;
    fn own_pn(&self) -> Option<Jid>;
    async fn namespace_alias(&self, peer: &Jid) -> Option<Jid>;
    async fn learn_identity(
        &self,
        lid: &Jid,
        pn: &Jid,
        offline: bool,
        has_username: bool,
    ) -> Result<(), CallError>;
    fn is_connected(&self) -> bool;
    fn generate_request_id(&self) -> String;
    async fn resolve_peer(&self, peer: &Jid) -> Result<Jid, CallError>;
    async fn prepare_keys(&self, peer: &Jid, plaintext: &[u8])
    -> Result<DirectCallKeys, CallError>;
    async fn decrypt(
        &self,
        peer: &Jid,
        enc_type: EncType,
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, CallError>;
    async fn read_token(&self, peer: &Jid) -> Result<Option<Vec<u8>>, CallError>;
    async fn issue_token(&self, peer: &Jid);
    async fn send_node(&self, node: Node) -> Result<(), CallError>;
    /// Register synchronously, before `send_node`, and unregister when the returned future drops.
    fn watch_ack(&self, id: &str) -> Result<CallAckFuture, CallError>;
}

pub type CallAckFuture = BoxFuture<'static, Result<Node, CallError>>;

/// Session state shared by call builders and inbound signaling on one host.
pub struct DirectCallState {
    epoch: portable_atomic::AtomicU64,
    /// Set by the host while the connection it is closing is expected to be replaced, and read by
    /// the cleanup that follows, which then suspends calls with live media instead of ending them.
    pub(crate) signaling_expected_back: portable_atomic::AtomicBool,
    pub(crate) call_registry: Arc<CallRegistry>,
    pub(crate) pending_outgoing_calls: Arc<Mutex<HashMap<String, super::facade::PendingOutgoing>>>,
    pub(crate) answer_transition_locks: OnceLock<[Arc<AsyncMutex<()>>; ANSWER_TRANSITION_LANES]>,
}

impl DirectCallState {
    pub fn new(registry: Arc<CallRegistry>) -> Self {
        Self {
            epoch: portable_atomic::AtomicU64::new(0),
            signaling_expected_back: portable_atomic::AtomicBool::new(false),
            call_registry: registry,
            pending_outgoing_calls: Arc::new(Mutex::new(HashMap::new())),
            answer_transition_locks: OnceLock::new(),
        }
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(portable_atomic::Ordering::Acquire)
    }

    pub(crate) fn ensure_epoch(&self, epoch: u64) -> Result<(), CallError> {
        if self.epoch() == epoch {
            Ok(())
        } else {
            Err(CallError::Connect(
                "connection changed during call setup".into(),
            ))
        }
    }

    pub fn registry(&self) -> &Arc<CallRegistry> {
        &self.call_registry
    }

    pub(crate) fn transition(&self, id: &str) -> Arc<AsyncMutex<()>> {
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hash);
        let locks = self
            .answer_transition_locks
            .get_or_init(|| std::array::from_fn(|_| Arc::new(AsyncMutex::new(()))));
        locks[hash.finish() as usize % locks.len()].clone()
    }
}

pub(crate) use super::direct_control::{announce_muted_locked, terminate_for_generation};

pub(crate) fn terminate_local_if_current(
    host: &dyn DirectCallHost,
    call_id: &str,
    generation: u64,
) {
    super::facade::terminate_direct_local(host, call_id, generation);
}

/// Start a direct PCM call using the host's existing connection and Signal sessions.
pub async fn call_pcm(
    host: Arc<dyn DirectCallHost>,
    peer: &Jid,
    source: Arc<dyn wacore::voip_control::AudioSource>,
    sink: Arc<dyn wacore::voip_control::AudioSink>,
) -> Result<super::CallHandle, CallError> {
    if !host.is_connected() {
        return Err(CallError::Connect("host is disconnected".into()));
    }
    super::facade::call_direct_pcm(host, peer, source, sink).await
}

/// Accept a retained direct offer. Media I/O stays on the same bounded backend.
pub async fn accept_pcm(
    host: Arc<dyn DirectCallHost>,
    incoming: &wacore::types::call::IncomingCall,
    source: Arc<dyn wacore::voip_control::AudioSource>,
    sink: Arc<dyn wacore::voip_control::AudioSink>,
) -> Result<super::CallHandle, CallError> {
    if !host.is_connected() {
        return Err(CallError::Connect("host is disconnected".into()));
    }
    super::facade::accept_direct_pcm(host, incoming, source, sink).await
}

/// Synchronous local cleanup; disconnect never sends stale signaling on another connection.
pub fn disconnect(host: &dyn DirectCallHost) {
    host.call_state()
        .epoch
        .fetch_add(1, portable_atomic::Ordering::AcqRel);
    host.call_state().call_registry.abort_all();
    super::facade::drain_direct_pending(host);
}

/// The signaling connection dropped and the host expects to reconnect: end what cannot outlive it
/// and let calls with live media keep running, for at most `resume_within`.
///
/// The epoch still advances, so anything begun before the loss (call setup, an answer in flight)
/// is fenced out as it is by [`disconnect`]. Kept calls are told with
/// [`CallEvent::SignalingSuspended`](wacore::voip_control::CallEvent::SignalingSuspended); the
/// host reports the new connection with [`resume_signaling`], and a call still waiting when the
/// deadline passes ends with
/// [`CallEvent::SignalingResumeTimedOut`](wacore::voip_control::CallEvent::SignalingResumeTimedOut).
/// A logout, a replaced stream or a ban is not this: the host calls [`disconnect`] for those.
pub fn suspend_signaling(
    host: &Arc<dyn DirectCallHost>,
    resume_within: Duration,
) -> wacore::voip_control::registry::SignalingSuspension {
    host.call_state()
        .epoch
        .fetch_add(1, portable_atomic::Ordering::AcqRel);
    let outcome = host.call_state().call_registry.suspend_signaling();
    super::facade::drain_direct_pending(&**host);
    if let Some(episode) = outcome.episode {
        let runtime = host.runtime();
        let host = Arc::downgrade(host);
        let sleeper = runtime.clone();
        runtime
            .spawn(Box::pin(async move {
                sleeper.sleep(resume_within).await;
                if let Some(host) = host.upgrade() {
                    host.call_state()
                        .call_registry
                        .end_suspended_signaling(episode);
                }
            }))
            .detach();
    }
    outcome
}

/// The host has a signaling connection again: re-attach every suspended call to it. Returns how
/// many calls had been waiting.
///
/// Nothing is rebuilt. A call's signaling is addressed by call id and sent through the host's
/// current connection, so once the host can send and receive again the call is as it was.
pub fn resume_signaling(host: &dyn DirectCallHost) -> usize {
    host.call_state().call_registry.resume_signaling()
}

pub use super::direct_ingress::ingest_direct_call as ingest;

/// Send termination through the same generation-fenced path as native call control.
pub async fn terminate(
    host: &dyn DirectCallHost,
    call_id: &str,
    peer: &Jid,
    creator: &Jid,
) -> Result<(), CallError> {
    super::direct_control::terminate(host, call_id, peer, creator).await
}
