use super::*;

fn doc(value: serde_json::Value) -> Document {
    match value {
        serde_json::Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

#[test]
fn filters_compare_numbers_strings_and_combine() {
    let d = doc(serde_json::json!({"year": 2024, "lang": "en", "draft": false}));
    assert!(Filter::cmp("year", Op::Gte, 2020).matches(&d));
    assert!(!Filter::cmp("year", Op::Lt, 2020).matches(&d));
    assert!(
        Filter::eq("lang", "en")
            .and(Filter::eq("draft", false))
            .matches(&d)
    );
    assert!(
        Filter::eq("lang", "fr")
            .or(Filter::any_of("lang", ["de", "en"]))
            .matches(&d)
    );
    assert!(Filter::Not(Box::new(Filter::eq("lang", "fr"))).matches(&d));
    assert!(!Filter::eq("missing", 1).matches(&d));
}

#[test]
fn filter_fields_must_be_identifiers() {
    assert!(Filter::eq("a_b1", 1).validate().is_ok());
    assert_eq!(
        Filter::eq("a'; drop", 1).validate().unwrap_err().kind(),
        ErrorKind::InvalidRequest
    );
}

#[test]
fn a_filter_round_trips_through_json() {
    let filter = Filter::eq("lang", "en").and(Filter::cmp("year", Op::Gt, 2.5));
    let json = serde_json::to_string(&filter).unwrap();
    assert_eq!(serde_json::from_str::<Filter>(&json).unwrap(), filter);
}

#[tokio::test]
async fn the_in_memory_store_ranks_by_similarity_and_deletes() {
    let store = InMemoryStore::default();
    store
        .upsert(vec![
            Record::new("near", vec![1.0, 0.0], serde_json::json!({"n": 1})),
            Record::new("far", vec![-1.0, 0.0], serde_json::json!({"n": 2})),
        ])
        .await
        .unwrap();
    let hits = store
        .search_ids(Query::new(vec![0.9, 0.1], 2))
        .await
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["near", "far"]
    );
    let above = store
        .search(Query::new(vec![1.0, 0.0], 2).with_min_score(0.5))
        .await
        .unwrap();
    assert_eq!(above.len(), 1);
    store.delete(vec!["near".into()]).await.unwrap();
    assert_eq!(
        store
            .search_ids(Query::new(vec![1.0, 0.0], 5))
            .await
            .unwrap()
            .len(),
        1
    );
}
