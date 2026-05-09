//! Chat Completions SSE → Responses API SSE 流式转换模块
//!
//! 实现 OpenAI Chat Completions SSE 格式到 Responses API SSE 命名事件格式的转换。
//!
//! Chat Completions SSE 使用简单的 delta chunk 模型:
//!   data: {"choices":[{"delta":{"content":"Hello"},...}]}
//!
//! Responses API SSE 使用命名事件 (named events) 的生命周期模型:
//!   response.created → output_item.added → content_part.added →
//!   output_text.delta → content_part.done → output_item.done → response.completed

use crate::proxy::sse::{strip_sse_field, take_sse_block};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
struct ChatChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, serde::Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatToolCallDelta>>,
}

#[derive(Debug, serde::Deserialize)]
struct ChatToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChatFunctionDelta>,
}

#[derive(Debug, serde::Deserialize)]
struct ChatFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallAgg {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct ChunkAgg {
    text_content: String,
    tool_calls: HashMap<usize, ToolCallAgg>,
    finish_reason: Option<String>,
}

/// 创建从 Chat Completions SSE 到 Responses API SSE 的转换流
pub fn create_responses_sse_stream_from_chat<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut response_id: Option<String> = None;
        let mut model: Option<String> = None;
        let mut msg_item_id: Option<String> = None;
        let mut agg = ChunkAgg::default();

        let mut sent_response_created = false;
        let mut sent_item_added = false;
        let mut sent_content_part_added = false;
        let mut completed = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::proxy::sse::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(block) = take_sse_block(&mut buffer) {
                        let block = block.trim().to_string();
                        if block.is_empty() {
                            continue;
                        }

                        let data_str = match block
                            .lines()
                            .find_map(|line| strip_sse_field(line, "data"))
                        {
                            Some(d) => d.trim().to_string(),
                            None => continue,
                        };

                        if data_str == "[DONE]" {
                            if !completed {
                                if sent_content_part_added {
                                    yield Ok(Bytes::from(build_completion_events(
                                        msg_item_id.as_deref().unwrap_or(""),
                                        &agg,
                                    )));
                                }
                                yield Ok(Bytes::from(build_response_completed(
                                    response_id.as_deref().unwrap_or(""),
                                    model.as_deref().unwrap_or(""),
                                    &agg,
                                    &json!({}),
                                )));
                                completed = true;
                            }
                            continue;
                        }

                        let chunk: ChatChunk = match serde_json::from_str(&data_str) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };

                        if !chunk.id.is_empty() {
                            response_id = Some(chunk.id.clone());
                        }
                        if !chunk.model.is_empty() {
                            model = Some(chunk.model.clone());
                        }
                        if msg_item_id.is_none() {
                            msg_item_id = Some(format!(
                                "msg_{}",
                                uuid::Uuid::new_v4().to_string().replace('-', "")
                            ));
                        }

                        if !sent_response_created {
                            yield Ok(Bytes::from(build_response_created_event(
                                response_id.as_deref().unwrap_or(""),
                                model.as_deref().unwrap_or(""),
                            )));
                            sent_response_created = true;
                        }

                        for choice in &chunk.choices {
                            if let Some(ref role) = choice.delta.role {
                                if role == "assistant" && !sent_item_added {
                                    yield Ok(Bytes::from(build_output_item_added(
                                        msg_item_id.as_deref().unwrap_or(""),
                                    )));
                                    sent_item_added = true;
                                }
                            }

                            if !sent_content_part_added
                                && choice.delta.content.is_some()
                                && sent_item_added
                            {
                                yield Ok(Bytes::from(build_content_part_added(
                                    msg_item_id.as_deref().unwrap_or(""),
                                )));
                                sent_content_part_added = true;
                            }

                            if let Some(ref text) = choice.delta.content {
                                if !text.is_empty() {
                                    agg.text_content.push_str(text);
                                    yield Ok(Bytes::from(build_output_text_delta(
                                        msg_item_id.as_deref().unwrap_or(""),
                                        text,
                                    )));
                                }
                            }

                            if let Some(ref tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let entry = agg.tool_calls.entry(tc.index).or_default();
                                    if let Some(ref id) = tc.id {
                                        entry.id = id.clone();
                                    }
                                    if let Some(ref func) = tc.function {
                                        if let Some(ref name) = func.name {
                                            entry.name = name.clone();
                                        }
                                        if let Some(ref args) = func.arguments {
                                            entry.arguments.push_str(args);
                                        }
                                    }
                                }
                            }

                            if let Some(ref reason) = choice.finish_reason {
                                agg.finish_reason = Some(reason.clone());
                            }
                        }

                        if let Some(ref usage) = chunk.usage {
                            if !completed {
                                if sent_content_part_added {
                                    yield Ok(Bytes::from(build_completion_events(
                                        msg_item_id.as_deref().unwrap_or(""),
                                        &agg,
                                    )));
                                }
                                yield Ok(Bytes::from(build_response_completed(
                                    response_id.as_deref().unwrap_or(""),
                                    model.as_deref().unwrap_or(""),
                                    &agg,
                                    usage,
                                )));
                                completed = true;
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        if !completed {
            if sent_content_part_added {
                yield Ok(Bytes::from(build_completion_events(
                    msg_item_id.as_deref().unwrap_or(""),
                    &agg,
                )));
            }
            yield Ok(Bytes::from(build_response_completed(
                response_id.as_deref().unwrap_or(""),
                model.as_deref().unwrap_or(""),
                &agg,
                &json!({}),
            )));
        }
    }
}

fn build_response_created_event(id: &str, model: &str) -> String {
    let response = json!({
        "id": id,
        "object": "response",
        "model": model,
        "status": "in_progress",
        "output": []
    });
    format!(
        "event: response.created\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.created",
            "response": response
        }))
        .unwrap_or_default()
    )
}

fn build_output_item_added(msg_id: &str) -> String {
    let item = json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": []
    });
    format!(
        "event: response.output_item.added\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item
        }))
        .unwrap_or_default()
    )
}

fn build_content_part_added(msg_id: &str) -> String {
    format!(
        "event: response.content_part.added\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.content_part.added",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": []
            }
        }))
        .unwrap_or_default()
    )
}

fn build_output_text_delta(msg_id: &str, delta: &str) -> String {
    format!(
        "event: response.output_text.delta\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.output_text.delta",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta
        }))
        .unwrap_or_default()
    )
}

/// 发送完成事件: content_part.done + output_item.done
fn build_completion_events(msg_id: &str, agg: &ChunkAgg) -> String {
    let status = match agg.finish_reason.as_deref() {
        Some("stop") | Some("content_filter") => "completed",
        Some("length") => "incomplete",
        Some("tool_calls") => "completed",
        _ => "completed",
    };

    let content_part_done = format!(
        "event: response.content_part.done\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": agg.text_content,
                "annotations": []
            }
        }))
        .unwrap_or_default()
    );

    let output_item_done = format!(
        "event: response.output_item.done\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [{
                    "type": "output_text",
                    "text": agg.text_content,
                    "annotations": []
                }]
            }
        }))
        .unwrap_or_default()
    );

    format!("{content_part_done}{output_item_done}")
}

fn build_response_completed(
    id: &str,
    model: &str,
    agg: &ChunkAgg,
    usage: &Value,
) -> String {
    let default_zero = json!(0);
    let input_tokens = usage.get("prompt_tokens").unwrap_or(&default_zero);
    let output_tokens = usage.get("completion_tokens").unwrap_or(&default_zero);
    let total_tokens = usage.get("total_tokens").unwrap_or(&default_zero);

    let status = match agg.finish_reason.as_deref() {
        Some("stop") | Some("content_filter") => "completed",
        Some("length") => "incomplete",
        Some("tool_calls") => "completed",
        _ => "completed",
    };

    let output: Vec<Value> = if agg.tool_calls.is_empty() {
        vec![json!({
            "type": "message",
            "id": format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', "")),
            "role": "assistant",
            "status": status,
            "content": [{
                "type": "output_text",
                "text": agg.text_content,
                "annotations": []
            }]
        })]
    } else {
        let mut items: Vec<Value> = Vec::new();
        if !agg.text_content.is_empty() {
            items.push(json!({
                "type": "message",
                "id": format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', "")),
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": agg.text_content,
                    "annotations": []
                }]
            }));
        }
        for tc in agg.tool_calls.values() {
            items.push(json!({
                "type": "function_call",
                "id": tc.id,
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "status": "completed"
            }));
        }
        items
    };

    let response = json!({
        "id": id,
        "object": "response",
        "model": model,
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
    });

    format!(
        "event: response.completed\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.completed",
            "response": response
        }))
        .unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_response_created_event() {
        let event = build_response_created_event("resp_123", "gpt-5.4");
        assert!(event.starts_with("event: response.created\n"));
        assert!(event.contains("resp_123"));
        assert!(event.contains("gpt-5.4"));
    }

    #[test]
    fn test_build_output_item_added() {
        let event = build_output_item_added("msg_abc");
        assert!(event.starts_with("event: response.output_item.added\n"));
        assert!(event.contains("msg_abc"));
    }

    #[test]
    fn test_build_output_text_delta() {
        let event = build_output_text_delta("msg_abc", "Hello");
        assert!(event.starts_with("event: response.output_text.delta\n"));
        assert!(event.contains("Hello"));
    }

    #[test]
    fn test_build_response_completed() {
        let agg = ChunkAgg {
            text_content: "Hello!".to_string(),
            tool_calls: HashMap::new(),
            finish_reason: Some("stop".to_string()),
        };
        let event = build_response_completed(
            "resp_123",
            "gpt-5.4",
            &agg,
            &json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        );
        assert!(event.starts_with("event: response.completed\n"));
        assert!(event.contains("input_tokens"));
        assert!(event.contains("output_tokens"));
    }
}