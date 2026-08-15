// should_compile: a real Normal task may await a spawned High typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{High, Normal},
    CompletionCell, PriorityClass,
};

static HIGH_RESULT: CompletionCell<u32> = CompletionCell::new();
static NORMAL_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let high = executor.spawn_typed(PriorityClass::<High>::new(), &HIGH_RESULT, |_| async { 7 })?;
    let _normal = executor.spawn_typed(
        PriorityClass::<Normal>::new(),
        &NORMAL_RESULT,
        move |normal| async move {
            let mut high = high;
            let _ = normal.wait_for(&mut high).await;
        },
    )?;
    Ok(())
}
