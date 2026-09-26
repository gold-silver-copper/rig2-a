use std::sync::Arc;

use super::*;

struct Shout;

impl Task for Shout {
    const NAME: &'static str = "shout";
    type Input = String;
    type Output = String;
    type Capabilities = ();
}

struct Upper(ModelInfo);

impl Model<Shout> for Upper {
    fn info(&self) -> &ModelInfo {
        &self.0
    }

    fn capabilities(&self) {}

    fn invoke(&self, input: String) -> BoxFuture<'static, Result<String>> {
        Box::pin(async move { Ok(input.to_uppercase()) })
    }
}

#[test]
fn an_erased_model_runs_with_converted_input() {
    let model: Arc<dyn Model<Shout>> = Arc::new(Upper(ModelInfo::new("local", "upper")));
    let out = futures::executor::block_on(model.run("hi")).unwrap();
    assert_eq!(out, "HI");
    assert_eq!(model.info().model, "upper");
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn a_call_future_is_static_and_send() {
    fn assert_spawnable<F: std::future::Future + Send + 'static>(_: F) {}
    let model = Upper(ModelInfo::new("local", "upper"));
    assert_spawnable(model.run("x"));
}
