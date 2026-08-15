//! Small RAII helpers for fallible host-side harness paths.

use std::sync::Arc;

use gpu_host::error::{GpuHostError, Result};
use gpu_host::hostcall::HostcallBuffer;

/// Owns the hostcall listener thread and always shuts it down before the
/// underlying mapped buffer can be dropped, including on an early `?` return.
pub(crate) struct HostcallListener {
    lifetime: ListenerLifetime,
    _buffer: Arc<HostcallBuffer>,
}

struct ListenerLifetime {
    shutdown: Box<dyn Fn() + Send + Sync>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl HostcallListener {
    pub(crate) fn start<F>(buffer: Arc<HostcallBuffer>, callback: F) -> Self
    where
        F: FnMut(&[u8]) + Send + 'static,
    {
        let listener_buffer = Arc::clone(&buffer);
        let handle = std::thread::spawn(move || listener_buffer.listen(callback));
        let shutdown_buffer = Arc::clone(&buffer);
        Self {
            lifetime: ListenerLifetime::new(move || shutdown_buffer.signal_shutdown(), handle),
            _buffer: buffer,
        }
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.lifetime.shutdown_and_join()
    }
}

impl ListenerLifetime {
    fn new(
        shutdown: impl Fn() + Send + Sync + 'static,
        handle: std::thread::JoinHandle<()>,
    ) -> Self {
        Self {
            shutdown: Box::new(shutdown),
            handle: Some(handle),
        }
    }

    fn shutdown_and_join(&mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        (self.shutdown)();
        handle.join().map_err(|_| GpuHostError::Verification {
            test: "hostcall_listener",
            detail: "listener thread panicked".to_string(),
        })
    }
}

impl Drop for ListenerLifetime {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown_and_join() {
            eprintln!("hostcall listener cleanup failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;

    fn fail_after_listener_start(
        shutdown_count: Arc<AtomicUsize>,
        worker_exited: Arc<AtomicBool>,
    ) -> Result<()> {
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            shutdown_rx.recv().expect("shutdown signal");
            worker_exited.store(true, Ordering::Release);
        });
        let _guard = ListenerLifetime::new(
            move || {
                shutdown_count.fetch_add(1, Ordering::AcqRel);
                shutdown_tx.send(()).expect("worker still waiting");
            },
            handle,
        );
        Err(GpuHostError::Verification {
            test: "listener_fault_injection",
            detail: "injected post-start failure".to_string(),
        })
    }

    #[test]
    fn early_error_signals_and_joins_listener_exactly_once() {
        let shutdown_count = Arc::new(AtomicUsize::new(0));
        let worker_exited = Arc::new(AtomicBool::new(false));

        let error =
            fail_after_listener_start(Arc::clone(&shutdown_count), Arc::clone(&worker_exited))
                .expect_err("fault injection must escape the guarded scope");

        assert!(error.to_string().contains("injected post-start failure"));
        assert_eq!(shutdown_count.load(Ordering::Acquire), 1);
        assert!(worker_exited.load(Ordering::Acquire));
    }

    #[test]
    fn explicit_finish_reports_listener_panic() {
        let handle = std::thread::spawn(|| panic!("injected listener panic"));
        let mut guard = ListenerLifetime::new(|| {}, handle);

        let error = guard
            .shutdown_and_join()
            .expect_err("listener panic must be observable");

        assert!(error.to_string().contains("listener thread panicked"));
    }
}
