use serde::Serialize;

use super::*;

#[derive(Debug, Serialize, serde::Deserialize, schemars::JsonSchema)]
enum Unit {
    Celsius,
    Fahrenheit,
}

#[derive(Debug, Serialize)]
struct Forecast {
    city: String,
    degrees: f64,
}

/// Look up the weather.
///
/// Returns the temperature.
#[rig2_agent::tool]
fn weather(#[describe("the city name")] city: String, unit: Option<Unit>) -> Result<Forecast> {
    if city.is_empty() {
        return Err(Error::new(ErrorKind::InvalidRequest, "no city"));
    }
    let degrees = if matches!(unit, Some(Unit::Fahrenheit)) {
        68.0
    } else {
        20.0
    };
    Ok(Forecast { city, degrees })
}

/// Say hello.
#[rig2_agent::tool]
fn greet(name: String) -> String {
    format!("hello {name}")
}

#[test]
fn the_macro_builds_a_definition_from_the_signature_and_docs() {
    let definition = Weather.definition();
    assert_eq!(definition.name, "weather");
    assert_eq!(
        definition.description,
        "Look up the weather.  Returns the temperature."
    );
    let props = &definition.parameters["properties"];
    assert_eq!(props["city"]["description"], "the city name");
    assert!(props.get("unit").is_some());
    assert_eq!(
        definition.parameters["required"],
        serde_json::json!(["city"])
    );
}

#[tokio::test]
async fn the_generated_tool_parses_arguments_and_serializes_the_output() {
    let out = Weather
        .call(
            serde_json::json!({"city": "Paris", "unit": "Fahrenheit"}),
            ToolContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        out,
        vec![ToolOutput::Json(
            serde_json::json!({"city": "Paris", "degrees": 68.0})
        )]
    );
    let text = Greet
        .call(serde_json::json!({"name": "Ada"}), ToolContext::default())
        .await
        .unwrap();
    assert_eq!(text, vec![ToolOutput::Text("hello Ada".into())]);
}

#[tokio::test]
async fn bad_arguments_and_tool_errors_are_errors_the_model_can_see() {
    let bad = Weather
        .call(serde_json::json!({"town": "Paris"}), ToolContext::default())
        .await
        .unwrap_err();
    assert_eq!(bad.kind(), ErrorKind::InvalidRequest);
    assert!(bad.message().contains("weather"));
    let failed = Weather
        .call(serde_json::json!({"city": ""}), ToolContext::default())
        .await
        .unwrap_err();
    assert_eq!(failed.message(), "no city");
}

#[test]
fn the_function_itself_still_works() {
    let greeting = greet("Grace".into());
    assert_eq!(greeting, "hello Grace");
}
