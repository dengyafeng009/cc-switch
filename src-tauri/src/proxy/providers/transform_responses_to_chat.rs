//! OpenAI Responses API ↔ Chat Completions 格式转换模块
//!
//! 实现 OpenAI Responses API 格式与 Chat Completions 格式之间的双向转换。
//! 用于 NewAPI 等只支持 /v1/chat/completions 接口的供应商，
//! 使得 Codex Desktop (使用 /v1/responses) 可以透明地使用这些供应商。
//!
//! ## 转换方向
//! - `responses_to_chat_completions`: 请求方向，Responses → Chat Completions
//! - `chat_completions_to_responses`: 响应方向，Chat Completions → Responses

use crate::proxy::error::ProxyError;
use serde_json::{json, Value};

/// OpenAI Responses 请求 → Chat Completions 请求
///
/// 主要转换：
/// - `instructions` → system role message 插入到 messages 开头
/// - `input[]` → `messages[]` (格式相近，但需要处理 content 格式差异)
/// - `max_output_tokens` → `max_tokens`
/// - `tools[].name/description/parameters` → `tools[].function.name/description/parameters`
/// - `reasoning.effort` → 透传 (Chat Completions 也支持)
/// - `text.format` → 透传 (Chat Completions response_format)
pub fn responses_to_chat_completions(body: Value) -> Result<Value, ProxyError> {
    let mut result = json!({});

    // model 直接透传
    if let Some(model) = body.get("model").and_then(|m| m.as_str()) {
        result["model"] = json!(model);
    }

    // messages 数组
    let mut messages: Vec<Value> = Vec::new();

    // instructions → system message
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": instructions
            }));
        }
    }

    // input[] → messages[]
    if let Some(input_items) = body.get("input").and_then(|v| v.as_array()) {
        for item in input_items {
            if let Some(msg) = convert_input_item_to_message(item) {
                messages.push(msg);
            }
        }
    }

    result["messages"] = json!(messages);

    // max_output_tokens → max_tokens
    if let Some(v) = body.get("max_output_tokens") {
        result["max_tokens"] = v.clone();
    }

    // temperature / top_p 直接透传
    for key in &["temperature", "top_p", "stream"] {
        if let Some(v) = body.get(key) {
            result[key] = v.clone();
        }
    }

    // tools 格式转换: Responses 格式 → Chat Completions 格式
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let chat_tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name").and_then(|v| v.as_str())?;
                let mut chat_tool = json!({
                    "type": "function",
                    "function": {
                        "name": name
                    }
                });
                if let Some(desc) = tool.get("description").and_then(|v| v.as_str()) {
                    chat_tool["function"]["description"] = json!(desc);
                }
                if let Some(params) = tool.get("parameters") {
                    chat_tool["function"]["parameters"] = params.clone();
                }
                Some(chat_tool)
            })
            .collect();
        if !chat_tools.is_empty() {
            result["tools"] = json!(chat_tools);
        }
    }

    // tool_choice 透传 (两个 API 格式相同)
    if let Some(tool_choice) = body.get("tool_choice") {
        result["tool_choice"] = tool_choice.clone();
    }

    // reasoning.effort → reasoning_effort (Chat Completions 也支持)
    if let Some(reasoning) = body.get("reasoning").and_then(|v| v.get("effort")).and_then(|v| v.as_str()) {
        result["reasoning_effort"] = json!(reasoning);
    }

    // text.format → response_format (Chat Completions 支持 response_format)
    if let Some(text_config) = body.get("text").and_then(|v| v.get("format")) {
        if let Some(format_type) = text_config.get("type").and_then(|v| v.as_str()) {
            if format_type == "json_schema" || format_type == "json_object" {
                result["response_format"] = text_config.clone();
            }
        }
    }

    // 透传其他 Chat Completions 兼容的字段
    for key in &["frequency_penalty", "presence_penalty", "seed", "stop"] {
        if let Some(v) = body.get(key) {
            result[key] = v.clone();
        }
    }

    Ok(result)
}

/// 将 Responses API input item 转换为 Chat Completions message
fn convert_input_item_to_message(item: &Value) -> Option<Value> {
    let item_type = item.get("type").and_then(|v| v.as_str())?;

    match item_type {
        "message" => {
            let raw_role = item.get("role").and_then(|v| v.as_str())?;
            // Chat Completions API doesn't accept "developer" role; map to "system"
            let role = if raw_role == "developer" { "system" } else { raw_role };
            let content = item.get("content")?;

            // content 可能是 string 或 array
            let chat_content = if let Some(text) = content.as_str() {
                json!(text)
            } else if let Some(parts) = content.as_array() {
                // Responses content parts → Chat Completions content parts
                let chat_parts: Vec<Value> = parts
                    .iter()
                    .filter_map(convert_content_part_to_chat)
                    .collect();
                if chat_parts.is_empty() {
                    json!("")
                } else if chat_parts.len() == 1 && chat_parts[0].get("type").and_then(|v| v.as_str()) == Some("text") {
                    // 单个 text 可以简化为字符串
                    chat_parts[0].get("text").cloned().unwrap_or(json!(""))
                } else {
                    json!(chat_parts)
                }
            } else {
                json!("")
            };

            Some(json!({
                "role": role,
                "content": chat_content
            }))
        }
        "function_call" => {
            // function_call item → assistant message with tool_calls
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            Some(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }]
            }))
        }
        "function_call_output" => {
            // function_call_output item → tool message
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let output = item.get("output").cloned().unwrap_or(json!(""));

            Some(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output
            }))
        }
        _ => {
            // 忽略不认识的 item type (如 reasoning 等内部类型)
            None
        }
    }
}

/// 将 Responses content part 转换为 Chat Completions content part
fn convert_content_part_to_chat(part: &Value) -> Option<Value> {
    let part_type = part.get("type").and_then(|v| v.as_str())?;

    match part_type {
        "input_text" | "output_text" => {
            let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Some(json!({
                "type": "text",
                "text": text
            }))
        }
        "input_image" => {
            // image_url 格式在两个 API 中相同
            if let Some(image_url) = part.get("image_url") {
                Some(json!({
                    "type": "image_url",
                    "image_url": image_url.clone()
                }))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// OpenAI Chat Completions 响应 → Responses API 响应
///
/// 主要转换：
/// - `choices[].message` → `output[]`
/// - `choices[].finish_reason` → 顶层 `status` 和 output item 状态
/// - `usage.prompt_tokens` → `usage.input_tokens`
/// - `usage.completion_tokens` → `usage.output_tokens`
pub fn chat_completions_to_responses(body: Value) -> Result<Value, ProxyError> {
    let mut result = json!({});

    // id: 保留原始 id
    if let Some(id) = body.get("id").and_then(|v| v.as_str()) {
        result["id"] = json!(id);
    }

    // object 设为 "response"
    result["object"] = json!("response");

    // model
    if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
        result["model"] = json!(model);
    }

    // created_at
    if let Some(created) = body.get("created") {
        result["created_at"] = created.clone();
    }

    // choices[].message → output[]
    let mut output: Vec<Value> = Vec::new();
    let mut status = "completed";

    if let Some(choices) = body.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            let finish_reason = choice
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("stop");

            // Map finish_reason to Responses status
            status = match finish_reason {
                "stop" | "content_filter" => "completed",
                "length" => "incomplete",
                "tool_calls" => {
                    // tool_calls 在 message 中处理
                    "completed"
                }
                _ => "completed",
            };

            if let Some(message) = choice.get("message") {
                let items = convert_chat_message_to_output(message)?;
                output.extend(items);
            }
        }
    }

    if output.is_empty() {
        // 如果没有 output items，至少返回一个空的 message
        output.push(json!({
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{
                "type": "output_text",
                "text": "",
                "annotations": []
            }]
        }));
    }

    result["output"] = json!(output);
    result["status"] = json!(status);

    // usage 转换
    if let Some(usage) = body.get("usage") {
        let default_zero = json!(0);
        let input_tokens = usage.get("prompt_tokens").unwrap_or(&default_zero);
        let output_tokens = usage
            .get("completion_tokens")
            .unwrap_or(&default_zero);
        let total_tokens = usage.get("total_tokens").unwrap_or(&default_zero);

        let mut resp_usage = json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        });

        // 透传缓存相关字段（如果有）
        if let Some(cache_read) = usage
            .get("prompt_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
        {
            resp_usage["input_tokens_details"] = json!({
                "cached_tokens": cache_read
            });
        }

        result["usage"] = resp_usage;
    }

    Ok(result)
}

/// 将 Chat Completions message 转换为 Responses API output item(s)
fn convert_chat_message_to_output(message: &Value) -> Result<Vec<Value>, ProxyError> {
    let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("assistant");

    // 检查是否有 tool_calls
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        // 有 tool_calls → 返回 function_call output items
        let mut items: Vec<Value> = Vec::new();
        for tc in tool_calls {
            if let Some(func) = tc.get("function") {
                let call_id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                items.push(json!({
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed"
                }));
            }
        }
        if !items.is_empty() {
            return Ok(items);
        }
        // tool_calls present but all empty → return early with empty vec
        return Ok(Vec::new());
    }

    // 普通消息内容
    let content = message.get("content");

    let content_parts = if let Some(text) = content.and_then(|v| v.as_str()) {
        // 纯文本内容
        json!([{
            "type": "output_text",
            "text": text,
            "annotations": []
        }])
    } else if let Some(parts) = content.and_then(|v| v.as_array()) {
        // 多部分内容
        let resp_parts: Vec<Value> = parts
            .iter()
            .filter_map(|part| {
                let part_type = part.get("type").and_then(|v| v.as_str())?;
                match part_type {
                    "text" => {
                        let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        Some(json!({
                            "type": "output_text",
                            "text": text,
                            "annotations": []
                        }))
                    }
                    "image_url" => Some(part.clone()),
                    _ => None,
                }
            })
            .collect();
        json!(resp_parts)
    } else if content.is_none() || content == Some(&json!(null)) {
        // 无内容 (如 tool_calls 消息，但 tool_calls 已在上方处理)
        json!([{
            "type": "output_text",
            "text": "",
            "annotations": []
        }])
    } else {
        json!([{
            "type": "output_text",
            "text": "",
            "annotations": []
        }])
    };

    Ok(vec![json!({
        "type": "message",
        "role": role,
        "status": "completed",
        "content": content_parts
    })])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_responses_to_chat_completions_simple() {
        let body = json!({
            "model": "gpt-5.4",
            "instructions": "You are helpful.",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "max_output_tokens": 1024,
            "temperature": 0.7,
            "stream": false
        });
        let result = responses_to_chat_completions(body).unwrap();
        assert_eq!(result["model"], "gpt-5.4");
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(result["messages"][0]["content"], "You are helpful.");
        assert_eq!(result["messages"][1]["role"], "user");
        assert_eq!(result["messages"][1]["content"], "Hello");
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["temperature"], 0.7);
    }

    #[test]
    fn test_responses_to_chat_completions_with_tools() {
        let body = json!({
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }]
        });
        let result = responses_to_chat_completions(body).unwrap();
        let tool = &result["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "get_weather");
        assert_eq!(tool["function"]["description"], "Get weather");
        assert!(tool["function"]["parameters"].is_object());
    }

    #[test]
    fn test_responses_to_chat_completions_with_function_call() {
        let body = json!({
            "model": "gpt-5.4",
            "input": [
                {"type": "message", "role": "user", "content": "Weather in Tokyo?"},
                {"type": "function_call", "call_id": "call_123", "name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"},
                {"type": "function_call_output", "call_id": "call_123", "output": "Sunny, 20C"}
            ]
        });
        let result = responses_to_chat_completions(body).unwrap();
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);

        // function_call → assistant message with tool_calls
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["arguments"], "{\"location\":\"Tokyo\"}");

        // function_call_output → tool message
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_123");
        assert_eq!(msgs[2]["content"], "Sunny, 20C");
    }

    #[test]
    fn test_chat_completions_to_responses_simple() {
        let body = json!({
            "id": "chatcmpl-123",
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let result = chat_completions_to_responses(body).unwrap();
        assert_eq!(result["id"], "chatcmpl-123");
        assert_eq!(result["object"], "response");
        assert_eq!(result["status"], "completed");
        assert_eq!(result["output"][0]["type"], "message");
        assert_eq!(result["output"][0]["role"], "assistant");
        assert_eq!(result["output"][0]["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_chat_completions_to_responses_with_tool_call() {
        let body = json!({
            "id": "chatcmpl-456",
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
        });
        let result = chat_completions_to_responses(body).unwrap();
        let output = &result["output"][0];
        assert_eq!(output["type"], "function_call");
        assert_eq!(output["name"], "get_weather");
        assert_eq!(output["arguments"], "{\"location\":\"Tokyo\"}");
    }

    #[test]
    fn test_responses_to_chat_completions_input_text_array_content() {
        let body = json!({
            "model": "gpt-5.4",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello world"}]
            }]
        });
        let result = responses_to_chat_completions(body).unwrap();
        assert_eq!(result["messages"][0]["content"], "Hello world");
    }

    #[test]
    fn test_responses_to_chat_completions_reasoning() {
        let body = json!({
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": "Think deeply"}],
            "reasoning": {"effort": "high"}
        });
        let result = responses_to_chat_completions(body).unwrap();
        assert_eq!(result["reasoning_effort"], "high");
    }
}