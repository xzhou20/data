from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from tqdm import tqdm

def find_subsequence(tensor, subsequence):
    tensor_view = tensor.unfold(dimension=0, size=len(subsequence), step=1)
    indices = (tensor_view == subsequence).all(dim=1).nonzero().flatten()
    start_idx = indices[-1].item()
    end_idx = start_idx + subsequence.squeeze().shape[0]
    return [start_idx, end_idx]

def convert_text_to_tensor(question, retrieval_context, tokenizer):
    example = """Here is an example to help you know the format.
#### Example
Question: What holiday-themed Pop-Tart flavor does Pop-Tarts playfully suggest on their Instagram, eliciting mixed reactions?
The answer is: Gingerbread.
#### Example End
"""
    prompt_with_retrieval_context  = f"Please answer the question based on the provided context and your own knowledge. Only include the answer in your response without any note or explanation, and try to be concise. {example}\n\n Paragraph:\n{retrieval_context}\n\n Question: {question}" 
    # prompt_without_retrieval_context = f"Please answer the question based on the provided context and your own knowledge. Only include the answer in your response without any note or explanation, and try to be concise. {example}\n\n Paragraph: No paragraph availiable. \n\nQuestion: {item['question']}"

    # process prompt with retrieval context
    messages_with_retrieval_context = [
        {"role": "user", "content": prompt_with_retrieval_context},
    ]
    text_with_retrieval = tokenizer.apply_chat_template(messages_with_retrieval_context, tokenize=False)
    text_with_retrieval = text_with_retrieval + 'The answer is: '
    inputs_with_retrieval = tokenizer.encode(text_with_retrieval, add_special_tokens=False, return_tensors='pt')
    # inputs_with_retrieval = inputs_with_retrieval.to(model.device)
    return inputs_with_retrieval
    


def generate_with_router_logits(model, input_ids, question_indices, max_new_token=128):
    decode_expert = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, output_router_logits=True)
    question_expert =[item.squeeze()[question_indices[0]:question_indices[1]].cpu().tolist() for item in outputs.router_logits]
    question_expert = torch.tensor(question_expert)
    question_expert = question_expert.permute(1,0,2).tolist()
    step_router_logits = [item.squeeze()[-1].cpu().tolist() for item in outputs.router_logits]
    decode_expert.append(step_router_logits)
    past_kv = outputs.past_key_values
    inp = outputs.logits[:, -1].argmax(dim=-1) # (bsz, 1)
    response = [inp.item()]
    with torch.no_grad():
        for i in range(max_new_token):
            inp = inp.view(1,1)
            step_outputs = model(input_ids=inp, use_cache=True, past_key_values=past_kv, output_router_logits=True)
            step_router_logits = [item[-1].cpu().tolist() for item in step_outputs.router_logits]
            past_kv = step_outputs.past_key_values
            inp = step_outputs.logits[:, -1].argmax(dim=-1)
            response.append(inp.item())
            decode_expert.append(step_router_logits)
            if inp.item()==2:
                break
    return decode_expert, question_expert, response


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='mistralai/Mixtral-8x22B-Instruct-v0.1')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--max_new_token', type=int, default=128)
    parser.add_argument('--doc_top_n', type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    doc_top_n = args.doc_top_n
    output_path = args.output_path
    max_new_token = args.max_new_token


    with open(data_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(item) for item in data]
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    all_router_logits = []
    skip_list = []

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            processed_data = f.readlines()
            processed_data = [eval(item) for item in processed_data]
            processed_id_list = [item['question_id'] for item in processed_data]
            skip_list += processed_id_list

    for item in tqdm(data):
        if item['question_id'] in skip_list:
            continue

        retrieval_result = item["context"][:doc_top_n]
        if isinstance(retrieval_result[0], str):
            evidences = [f"[{i+1}] {context}" for i, context in enumerate(retrieval_result)] 
        else:
            evidences = [f"[{i+1}] {context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}" for i, context in enumerate(retrieval_result)] 
        retrieval_context = "\n".join(evidences)


        inputs_with_retrieval = convert_text_to_tensor(item['question'], retrieval_context, tokenizer).to(model.device)
        inputs_without_retrieval = convert_text_to_tensor(item['question'], 'No paragraph availiable.', tokenizer).to(model.device)
        question_ids = tokenizer.encode(' Question: '+item['question'], add_special_tokens=False, return_tensors='pt')[0].to(model.device)
        question_indices_with_retrieval = find_subsequence(inputs_with_retrieval[0], question_ids)
        question_indices_without_retrieval = find_subsequence(inputs_without_retrieval[0], question_ids)
        decode_expert_logits_with_retrieval, question_expert_logits_with_retrieval, response_with_retrieval = generate_with_router_logits(model, inputs_with_retrieval, question_indices_with_retrieval, max_new_token)
        decode_expert_logits_without_retrieval, question_expert_logits_without_retrieval, response_without_retrieval = generate_with_router_logits(model, inputs_without_retrieval, question_indices_without_retrieval, max_new_token)
        
        response_with_retrieval = tokenizer.convert_ids_to_tokens(response_with_retrieval)
        response_without_retrieval = tokenizer.convert_ids_to_tokens(response_without_retrieval)

        if 'ground_truth' in item:
            answer = item['ground_truth']
        else:
            answer = item['answer']

        save_item = {
            'question_id': item['question_id'],
            'question': item['question'],
            'answer': answer,
            'response_with_retrieval': response_with_retrieval,
            'response_without_retrieval': response_without_retrieval,
            'question_expert_logits_with_retrieval': question_expert_logits_with_retrieval,
            'question_expert_logits_without_retrieval': question_expert_logits_without_retrieval,
            'decode_expert_logits_with_retrieval': decode_expert_logits_with_retrieval,
            'decode_expert_logits_without_retrieval': decode_expert_logits_without_retrieval
        }    
        import json
        with open(output_path, 'a') as f:
            json.dump(save_item, f)
            f.write('\n')
