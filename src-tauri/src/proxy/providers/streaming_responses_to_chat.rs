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
//!   工具调用: output_item.added(function_call) → function_call_arguments.delta →
//!   function_call_arguments.done → output_item.done

use crate::proxy::sse::{strip_sse_field, take_sse_block};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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

/// Per-tool-call streaming state for emitting Responses API events.
#[derive(Debug, Default)]
struct ToolCallStreamState {
    id: String,
    name: String,
    item_id: String,
    arguments: String,
    sent_item_added: bool,
    sent_arguments_done: bool,
}

#[derive(Debug, Default)]
struct ChunkAgg {
    text_content: String,
    tool_calls: BTreeMap<usize, ToolCallStreamState>,
    finish_reason: Option<String>,
}

/// 生成新的 UUID-based ID（无连字符）
fn new_uuid_id() -> String {
    uuid::Uuid::new_v4().to_string().replace('-', "")
}

/// 关闭文本输出的完成事件: content_part.done + output_item.done
fn finalize_text_item(msg_id: &str, text_content: &str, finish_reason: Option<&str>) -> String {
    let status = match finish_reason {
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
                "text": text_content,
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
                    "text": text_content,
                    "annotations": []
                }]
            }
        }))
        .unwrap_or_default()
    );

    format!("{content_part_done}{output_item_done}")
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
        let mut text_item_finalized = false;
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
                                yield yield_completion_events(
                                    &mut agg, &mut text_item_finalized,
                                    &mut sent_content_part_added,
                                    msg_item_id.as_deref().unwrap_or(""),
                                    response_id.as_deref().unwrap_or(""),
                                    model.as_deref().unwrap_or(""),
                                    &json!({}),
                                );
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
                            msg_item_id = Some(new_uuid_id());
                        }

                        if !sent_response_created {
                            yield Ok(Bytes::from(build_response_created_event(
                                response_id.as_deref().unwrap_or(""),
                                model.as_deref().unwrap_or(""),
                            )));
                            sent_response_created = true;
                        }

                        for choice in &chunk.choices {
                            // 工具调用处理 — 优先检查，因为可能和文本同时出现
                            if let Some(ref tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    // Phase 1: update entry state (entry borrow scope)
                                    let (is_new_call, item_id, call_id, name, args_delta) = {
                                        let entry = agg.tool_calls.entry(tc.index).or_insert_with(|| {
                                            ToolCallStreamState {
                                                item_id: new_uuid_id(),
                                                ..Default::default()
                                            }
                                        });

                                        if let Some(ref id) = tc.id {
                                            entry.id = id.clone();
                                        }
                                        if let Some(ref func) = tc.function {
                                            if let Some(ref n) = func.name {
                                                entry.name = n.clone();
                                            }
                                            if let Some(ref args) = func.arguments {
                                                entry.arguments.push_str(args);
                                            }
                                        }

                                        let is_new = !entry.sent_item_added
                                            && !entry.id.is_empty()
                                            && !entry.name.is_empty();
                                        let item_id = entry.item_id.clone();
                                        let call_id = entry.id.clone();
                                        let name = entry.name.clone();

                                        if is_new {
                                            entry.sent_item_added = true;
                                        }

                                        let args_delta = tc.function.as_ref()
                                            .and_then(|f| f.arguments.as_deref())
                                            .filter(|a| !a.is_empty())
                                            .map(|a| a.to_string());

                                        (is_new, item_id, call_id, name, args_delta)
                                    };
                                    // Entry borrow dropped here

                                    // Phase 2: emit events using collected info
                                    if is_new_call {
                                        if sent_content_part_added && !text_item_finalized {
                                            let fc = agg.finish_reason.clone();
                                            yield Ok(Bytes::from(finalize_text_item(
                                                msg_item_id.as_deref().unwrap_or(""),
                                                &agg.text_content,
                                                fc.as_deref(),
                                            )));
                                            text_item_finalized = true;
                                        }

                                        yield Ok(Bytes::from(build_function_call_item_added(
                                            &item_id,
                                            &call_id,
                                            &name,
                                        )));
                                    }

                                    if let Some(ref delta) = args_delta {
                                        yield Ok(Bytes::from(
                                            build_function_call_arguments_delta(&item_id, delta),
                                        ));
                                    }
                                }
                            }

                            // 文本内容处理
                            if let Some(ref text) = choice.delta.content {
                                if !text.is_empty() {
                                    // 确保 output_item.added 和 content_part.added 已发送
                                    if !sent_item_added {
                                        yield Ok(Bytes::from(build_output_item_added(
                                            msg_item_id.as_deref().unwrap_or(""),
                                        )));
                                        sent_item_added = true;
                                    }
                                    if !sent_content_part_added {
                                        yield Ok(Bytes::from(build_content_part_added(
                                            msg_item_id.as_deref().unwrap_or(""),
                                        )));
                                        sent_content_part_added = true;
                                    }

                                    agg.text_content.push_str(text);
                                    yield Ok(Bytes::from(build_output_text_delta(
                                        msg_item_id.as_deref().unwrap_or(""),
                                        text,
                                    )));
                                }
                            }

                            if let Some(ref role) = choice.delta.role {
                                if role == "assistant" && !sent_item_added && agg.tool_calls.is_empty() {
                                    yield Ok(Bytes::from(build_output_item_added(
                                        msg_item_id.as_deref().unwrap_or(""),
                                    )));
                                    sent_item_added = true;
                                }
                            }

                            if let Some(ref reason) = choice.finish_reason {
                                agg.finish_reason = Some(reason.clone());
                            }
                        }

                        if let Some(ref usage) = chunk.usage {
                            if !completed {
                                yield yield_completion_events(
                                    &mut agg, &mut text_item_finalized,
                                    &mut sent_content_part_added,
                                    msg_item_id.as_deref().unwrap_or(""),
                                    response_id.as_deref().unwrap_or(""),
                                    model.as_deref().unwrap_or(""),
                                    usage,
                                );
                                completed = true;
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        if !completed {
            yield yield_completion_events(
                &mut agg, &mut text_item_finalized,
                &mut sent_content_part_added,
                msg_item_id.as_deref().unwrap_or(""),
                response_id.as_deref().unwrap_or(""),
                model.as_deref().unwrap_or(""),
                &json!({}),
            );
        }
    }
}

/// 产出完成阶段所有事件：文本 item 关闭、工具调用 item 关闭、response.completed
fn yield_completion_events(
    agg: &mut ChunkAgg,
    text_item_finalized: &mut bool,
    sent_content_part_added: &mut bool,
    msg_item_id: &str,
    response_id: &str,
    model: &str,
    usage: &Value,
) -> Result<Bytes, std::io::Error> {
    let mut events = String::new();

    // 关闭文本 item
    if *sent_content_part_added && !*text_item_finalized {
        events.push_str(&finalize_text_item(msg_item_id, &agg.text_content, agg.finish_reason.as_deref()));
        *text_item_finalized = true;
    }

    // 关闭所有工具调用 item
    for tc in agg.tool_calls.values_mut() {
        if tc.sent_item_added && !tc.sent_arguments_done {
            events.push_str(&format!(
                "event: response.function_call_arguments.done\ndata: {}\n\n",
                serde_json::to_string(&json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": tc.item_id,
                    "output_index": 0,
                    "arguments": tc.arguments
                }))
                .unwrap_or_default()
            ));
            events.push_str(&format!(
                "event: response.output_item.done\ndata: {}\n\n",
                serde_json::to_string(&json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": tc.item_id,
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "status": "completed"
                    }
                }))
                .unwrap_or_default()
            ));
            tc.sent_arguments_done = true;
        }
    }

    events.push_str(&build_response_completed(response_id, model, agg, usage));

    Ok(Bytes::from(events))
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

fn build_function_call_item_added(item_id: &str, call_id: &str, name: &str) -> String {
    let item = json!({
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "status": "in_progress"
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

fn build_function_call_arguments_delta(item_id: &str, delta: &str) -> String {
    format!(
        "event: response.function_call_arguments.delta\ndata: {}\n\n",
        serde_json::to_string(&json!({
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": 0,
            "delta": delta
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
            "id": new_uuid_id(),
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
        // 文本 item（如果有）
        if !agg.text_content.is_empty() {
            items.push(json!({
                "type": "message",
                "id": new_uuid_id(),
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": agg.text_content,
                    "annotations": []
                }]
            }));
        }
        // 工具调用 items（按 index 排序）
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
    use std::collections::BTreeMap;

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
    fn test_build_function_call_item_added() {
        let event = build_function_call_item_added("fc_123", "call_456", "get_weather");
        assert!(event.starts_with("event: response.output_item.added\n"));
        assert!(event.contains("fc_123"));
        assert!(event.contains("call_456"));
        assert!(event.contains("function_call"));
        assert!(event.contains("get_weather"));
    }

    #[test]
    fn test_build_function_call_arguments_delta() {
        let event = build_function_call_arguments_delta("fc_123", "{\"loc");
        assert!(event.starts_with("event: response.function_call_arguments.delta\n"));
        assert!(event.contains("{\\\"loc"));
    }

    #[test]
    fn test_build_response_completed_text_only() {
        let agg = ChunkAgg {
            text_content: "Hello!".to_string(),
            tool_calls: BTreeMap::new(),
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

    #[test]
    fn test_build_response_completed_with_tool_calls() {
        let mut tool_calls = BTreeMap::new();
        tool_calls.insert(
            0,
            ToolCallStreamState {
                id: "call_abc".to_string(),
                name: "get_weather".to_string(),
                item_id: "fc_001".to_string(),
                arguments: "{\"location\":\"Tokyo\"}".to_string(),
                sent_item_added: true,
                sent_arguments_done: true,
            },
        );
        let agg = ChunkAgg {
            text_content: String::new(),
            tool_calls,
            finish_reason: Some("tool_calls".to_string()),
        };
        let event = build_response_completed(
            "resp_456",
            "gpt-5.4",
            &agg,
            &json!({"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}),
        );
        assert!(event.starts_with("event: response.completed\n"));
        assert!(event.contains("function_call"));
        assert!(event.contains("get_weather"));
        assert!(event.contains("call_abc"));
    }
}