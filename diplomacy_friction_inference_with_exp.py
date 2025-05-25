import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import time
import pickle
 
import gc
 
from tqdm import tqdm
import logging
from datetime import datetime
from datasets import load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig,PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
# some experiments for the Llama model on safety
import os
import sys
import socket
import re
import random
import numpy as np
import torch
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import re

def parse_tags_robust(text, tags=None):
    """
    Enhanced parsing function that handles XML-style tags with improved robustness 
    for Diplomacy-specific content.
    
    Args:
        text (str): The text to parse.
        tags (list): List of tags to parse. If None, defaults to standard tags.
        
    Returns:
        dict: Dictionary with tag names as keys and lists of extracted content as values.
    """
    if tags is None:
        tags = ["friction", "rationale", "belief_state"]
    
    # Initialize result
    result = {tag: [] for tag in tags}
    
    # Tag mapping for variations
    tag_variations = {
        "belief_state": ["belief_state", "belief state", "beliefstate", "belief-state", "b"],
        "rationale": ["rationale", "rational", "reasoning", "reason", "r"],
        "friction": ["friction", "intervention", "advice", "f"]
    }
    
    # Reverse mapping for normalization
    tag_normalization = {}
    for normalized, variations in tag_variations.items():
        for variation in variations:
            tag_normalization[variation.lower()] = normalized
    
    # First attempt: Try exact XML-style tags
    for tag in tags:
        # Try variations of the tag name
        possible_tags = [tag]
        if tag in tag_variations:
            possible_tags.extend(tag_variations[tag])
        
        found = False
        for tag_var in possible_tags:
            # Try with and without underscores/hyphens
            tag_patterns = [
                tag_var,
                tag_var.replace("_", " "),
                tag_var.replace("-", " "),
                tag_var.replace(" ", "_"),
                tag_var.replace(" ", "-")
            ]
            
            for t_pattern in tag_patterns:
                # Match both <tag> and <tag></tag> patterns
                open_tag_pattern = f"<{t_pattern}>"
                close_tag_pattern = f"</{t_pattern}>"
                open_only_pattern = re.escape(open_tag_pattern) + r"(.*?)(?=" + re.escape(close_tag_pattern) + r"|\Z)"
                
                matches = re.findall(open_only_pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    result[tag].extend([m.strip() for m in matches])
                    found = True
                    break
            
            if found:
                break
    
    # Second attempt: Try markdown-style headings and formatting
    for tag in tags:
        if not result[tag]:  # Only if we didn't find anything in first pass
            possible_tags = [tag]
            if tag in tag_variations:
                possible_tags.extend(tag_variations[tag])
            
            for tag_var in possible_tags:
                # Various markdown and formatting patterns
                patterns = [
                    # Markdown headings
                    r'#{1,3}\s*' + re.escape(tag_var) + r'[:\s]*\n(.*?)(?=\n#|\n\n|\Z)',
                    # Bold with asterisks
                    r'\*\*\s*' + re.escape(tag_var) + r'[:\s]*\*\*\s*\n(.*?)(?=\n\*\*|\n\n|\Z)',
                    # Bold with underscores
                    r'__\s*' + re.escape(tag_var) + r'[:\s]*__\s*\n(.*?)(?=\n__|\n\n|\Z)',
                    # Plain text with colon
                    r'\b' + re.escape(tag_var) + r'[:\s]+(.*?)(?=\n\n|\Z)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    if matches:
                        result[tag].extend([m.strip() for m in matches])
                        break
    
    # Third attempt: Content-specific extraction based on sentence patterns
    for tag in tags:
        if not result[tag]:  # Only if we still didn't find anything
            if tag == "friction" or tag in tag_variations.get("friction", []):
                # Look for sentences that suggest a friction intervention
                question_patterns = [
                    # Questions about strategy
                    r'(?:Have you considered|Did you think about|What if|Could you|Why not).*?\?',
                    # Suggestions phrased as questions
                    r'(?:Wouldn\'t it be|Might it be|Isn\'t there).*?\?',
                ]
                
                # Extract any matching sentences
                for pattern in question_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Take the first 2 matching questions as friction
                        combined = " ".join(matches[:2]).strip()
                        if combined:
                            result[tag].append(combined)
                            break
    
    # Fourth attempt: Fall back to content structure analysis
    for tag in tags:
        if not result[tag]:  # If still nothing found
            # Split by double newlines to get paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if tag == "friction" and paragraphs:
                if len(paragraphs) >= 3:
                    # Friction is often at the end
                    result[tag].append(paragraphs[-1])
                else:
                    # Just use the last paragraph
                    result[tag].append(paragraphs[-1])
            
            elif tag == "rationale" and paragraphs:
                if len(paragraphs) >= 3:
                    # Rationale is often in the middle
                    result[tag].append(paragraphs[len(paragraphs)//2])
                elif len(paragraphs) == 2:
                    # Use first paragraph
                    result[tag].append(paragraphs[0])
            
            elif tag == "belief_state" and paragraphs:
                # Belief state is often at the beginning
                result[tag].append(paragraphs[0])
    
    # Final cleanup - remove quotes or special formatting from content
    for tag in result:
        result[tag] = [
            re.sub(r'^["\']\s*|\s*["\']$', '', item)  # Remove surrounding quotes
            for item in result[tag]
        ]
    
    return result


def handle_friction_logic(text):
    """
    Extract a meaningful friction intervention from text even when proper tags are missing.
    Improved to handle Diplomacy-specific content patterns.
    
    Args:
        text (str): The input text
        
    Returns:
        str: The extracted friction intervention
    """
    # First check if there's already a well-formatted friction section
    friction_match = re.search(r'<friction>(.*?)(?:</friction>|\Z)', text, re.DOTALL | re.IGNORECASE)
    if friction_match:
        return friction_match.group(1).strip()
    
    # Look for quotes that might be friction interventions
    quote_match = re.search(r'["\'](.*?)["\']', text, re.DOTALL)
    if quote_match and ("?" in quote_match.group(1) or "consider" in quote_match.group(1).lower()):
        return quote_match.group(1).strip()
    
    # Look for questions that might be friction interventions
    questions = re.findall(r'[^.!?]*\?', text)
    strategic_questions = [q for q in questions if any(term in q.lower() for term in 
                                                   ["strategy", "consider", "think", "reflect", 
                                                    "alternative", "option", "might", "could", 
                                                    "why not", "what if", "balance"])]
    if strategic_questions and len(strategic_questions) >= 1:
        return " ".join(strategic_questions[:2]).strip()
    
    # Look for sentences that have Diplomacy-specific strategy terms
    strategy_terms = ["support", "attack", "move", "defend", "ally", "alliance", "secure", 
                      "position", "flank", "threat", "opportunity", "risk", "coordinate"]
    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
    strategic_sentences = [s for s in sentences if any(term in s.lower() for term in strategy_terms)]
    
    if strategic_sentences:
        if len(strategic_sentences) >= 2:
            return f"{strategic_sentences[0]} {strategic_sentences[-1]}"
        else:
            return strategic_sentences[0]
    
    # Fall back to the original logic if nothing better is found
    if sentences:
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        else:
            return " ".join(sentences)
    
    return ""


def process_diplomacy_for_friction(test_data):

    processed_messages = []  # Create a list to store all processed messages
    count = 0
    for message in test_data:
        count = count+1
        sender = message['sender']
        recipient = message['recipient']

        if message['recipient'] == 'ALL' or message['recipient'] == message['sender'] or message["phase"].endswith('A') or message["phase"].endswith('R'):
            continue
        else:
            #board states format
            formatted_board = ""
            for country, units in message['units'].items():
                formatted_board += f"{country}: {', '.join(units)}\n"
            formatted_board = formatted_board.strip()
            #predicted orders for opponent
            opponent_orders = ""
            for country, units in message['predicted_orders'][message['recipient']].items():
                opponent_orders += f"{country}: {', '.join(units)}\n"
            formatted_opponent_orders = opponent_orders.strip()
            #message history
            message_history = ""
            start_index = max(0, len(message['prev_5_message']) - 5)
            for msg_info in message['prev_5_message'][start_index:]:
                sender = msg_info['sender']
                text = msg_info['message']
                message_history += f"Message from {sender}: {text}\n"
            message_history += f"Message from {message['sender']}: {message['message']}"
            #recommended orders for myself.
            recommended_orders = message['predicted_orders'][message['sender']][message['recipient']]
            formatted_recommended_orders = ",\n".join(recommended_orders)

                # Modify the prompts for friction interventions
            Prompt_level1 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert assistant specializing in the Diplomacy board game and collaborative reasoning. Your role is to analyze:
        1. The current board state.
        2. The recommended orders for the player.

        Your goal is to identify potential friction points in the player's strategy and provide a constructive intervention:

        <belief_state>
        Identify contradictions, oversights, or strategic misalignments in the recommended orders compared to the board state.
        </belief_state>

        <rationale>
        Explain why an intervention is needed—what's misaligned, its potential impact, and how alternative thinking could benefit the player.
        </rationale>

        <friction>
        Provide a concise friction intervention (max 2 sentences) that prompts self-reflection about their strategy. The first sentence should directly address the strategic concern, and the second should offer a reflective question or alternative perspective.
        </friction>
        Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        **Board State:**  
        {formatted_board}

        **Recommended Orders:**  
        {formatted_recommended_orders}

        **Advice Request:**  
        You are advising the player controlling {message['recipient']}. Identify potential friction points and provide constructive intervention.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

            Prompt_level2 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert assistant specializing in the Diplomacy board game and collaborative reasoning. Your role is to analyze:
        1. The current board state.
        2. The recommended orders for the player.
        3. The potential orders for every power.
        

        Your goal is to identify potential friction points in the player's strategy and provide a constructive intervention:

        <belief_state>
        Identify contradictions, oversights, or strategic misalignments in the recommended orders compared to the board state and other powers' likely moves.
        </belief_state>

        <rationale>
        Explain why an intervention is needed—what's misaligned, its potential impact, and how alternative thinking could benefit the player.
        </rationale>

        <friction>
        Provide a concise friction intervention (max 2 sentences) that prompts self-reflection about their strategy. The first sentence should directly address the strategic concern, and the second should offer a reflective question or alternative perspective.
        </friction>
        Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        **Board State:**  
        {formatted_board}

        **Recommended Orders for {message['recipient']}:**  
        {formatted_recommended_orders}

        **Potential Orders for other powers:**
        {formatted_opponent_orders}

        **Advice Request:**  
        You are advising the player controlling {message['recipient']}. Identify potential friction points and provide constructive intervention.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

            Prompt_level3 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert assistant specializing in the Diplomacy board game and collaborative reasoning. Your role is to analyze:
        1. The current board state.
        2. The recommended orders for the player.
        3. The potential orders for every power.
        4. The message history between the player and other players.

        Your goal is to identify potential friction points in the player's strategy and diplomatic approach, then provide a constructive intervention:

        <belief_state>
        Identify contradictions, oversights, or misalignments in the recommended orders compared to the board state, other powers' likely moves, and diplomatic communications.
        </belief_state>

        <rationale>
        Explain why an intervention is needed—what's misaligned in strategy or diplomacy, its potential impact, and how alternative thinking could benefit the player.
        </rationale>

        <friction>
        Provide a concise friction intervention (max 2 sentences) that prompts self-reflection about their strategy or diplomatic approach. The first sentence should directly address the concern, and the second should offer a reflective question or alternative perspective.
        </friction>
        Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        **Board State:**  
        {formatted_board}

        **Recommended Orders for {message['recipient']}:**  
        {formatted_recommended_orders}

        **Potential Orders for other powers:**
        {formatted_opponent_orders}

        **Message History:**
        {message_history}

        **Advice Request:**  
        You are advising the player controlling {message['recipient']}. Identify potential friction points and provide constructive intervention.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
            
            processed_message = message.copy()
         # Add the prompts to the copied message
            processed_message['Prompt_level1'] = Prompt_level1
            processed_message['Prompt_level2'] = Prompt_level2
            processed_message['Prompt_level3'] = Prompt_level3
             # Add this processed message to our list
            processed_messages.append(processed_message)

    return processed_messages

def generate_multiple_sequences_with_intrinsic_metrics(model, tokenizer, prompts, generation_args, device, 
                                                       strategy="beam_search", batched=False, 
                                                       reward_model=None, best_of_n=None, top_k_candidates=1, rm_tokenizer = None
                                                      , rm_max_length = None):
    """
    Generate multiple sequences using various strategies including best-of-N sampling.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        prompts: Input prompts
        generation_args: Arguments for generation
        device: Device to place tensors on
        strategy: Generation strategy ("beam_search", "top_k_sampling", "top_p_sampling", or "best_of_n")
        batched: Whether inputs are batched
        reward_model: Reward model for scoring in best-of-N sampling (AutoModelForSequenceClassification)
        best_of_n: Number of samples to generate for best-of-N sampling (default: None)
        top_k_candidates: Number of top candidates to return from best-of-N sampling (default: 1)
        
    Returns:
        generated_texts: List of generated texts
        all_generated_texts: List of all generated texts
    """
    if batched:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'

        cleaned_prompts = prompts.replace("\n", " ")  
        inputs = tokenizer(cleaned_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    else:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Handle best-of-N sampling strategy
    if strategy == "best_of_n":
        if reward_model is None:
            raise ValueError("Reward model must be provided for best-of-N sampling")
        
        if best_of_n is None or best_of_n <= 0:
            best_of_n = 4  # Default sample size
        
        with torch.no_grad():
            # Generate multiple candidates for each prompt
            all_candidates = []
            all_prompt_candidates = []
            
            # Use top_p or top_k sampling to generate diverse candidates
            sampling_strategy = generation_args.get("sampling_strategy", "top_p_sampling")
            # print("BON sampling strategy", sampling_strategy)
            for _ in range(best_of_n):
                if sampling_strategy == "top_p_sampling": 
                    print("RUNNING TopP sampling for BON")
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_p=generation_args.get("top_p", 0.9),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                else:  # Default to top_k_sampling
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_k=generation_args.get("top_k", 50),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                
                # Process the generated sequence
                for i in range(len(outputs.sequences)):
                    sequence = outputs.sequences[i]
                    prompt_length = input_ids.shape[-1]
                    new_tokens = sequence[prompt_length:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    if len(all_candidates) <= i:
                        all_candidates.append([])
                    
                    all_candidates[i].append(generated_text)
            
            # Score candidates with the reward model
            best_candidates = []
            all_prompt_candidates = []
            parsed_candidates = []
            tags_for_parsing = ["friction", "rationale", "t", "b"]  
            for candidate_index, candidates in enumerate(all_candidates):
                print(f"\n=== Processing candidates for prompt {candidate_index} ===")
                print(f"Number of candidates: {len(candidates)}")

                #parse the generated model outputs to get the friction + rationale

                        
                for candidate in candidates:
                    parsed_frictive_states_and_friction = parse_tags_robust(candidate, tags_for_parsing)
                    friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('friction', []))
                    if not friction_intervention:
                        friction_intervention = handle_friction_logic(candidate)

                    rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                    friction_and_rationale = rationale + friction_intervention
                    parsed_candidates.append(friction_and_rationale)
                    # print("PARSED friction + rationale",friction_and_rationale )
                # For each candidate, prepare input for reward model
                candidate_inputs = [prompts + " " + f"</s> {candidate} </s>" for candidate in parsed_candidates]
                tokenized_inputs = rm_tokenizer(candidate_inputs, return_tensors="pt", padding=True, truncation=True, max_length=rm_max_length).to(device)
                
                # Get scores from reward model
                reward_outputs = reward_model(**tokenized_inputs) 
                scores = reward_outputs.logits.squeeze(-1)
                
                ## Print all candidates with their scores
                print("\nAll candidates with scores:")
                for i, (candidate, score) in enumerate(zip(candidates, scores)):
                    print(f"Candidate {i}: Score = {score:.4f}")
                    print(f"Text snippet: {candidate[:50]}...")
                
                # Get top-k indices
                if top_k_candidates > len(candidates):
                    top_k_candidates = len(candidates)
                
                # Fix the error by converting bfloat16 to float32 before calling numpy()
                top_result = torch.topk(scores, top_k_candidates)
                top_indices = top_result.indices.cpu().numpy()
                top_values = top_result.values.cpu().float().numpy()  # Convert to float32 first
                
                print(f"\nTop {top_k_candidates} candidates:")
                # Print only the top-k candidates
                for rank, (idx, score) in enumerate(zip(top_indices, top_values)):
                    print(f"Rank {rank+1}: Candidate {idx} with score {score:.4f}")
                    print(f"Text: {candidates[idx][:300]}...")
                
                # Store the chosen candidates for verification
                prompt_best_candidates = [candidates[idx] for idx in top_indices]
                
                # Verification check
                max_score_idx = scores.argmax().item()
                if max_score_idx != top_indices[0]:
                    print(f"WARNING: Discrepancy detected! argmax={max_score_idx} but topk.indices[0]={top_indices[0]}")
                else:
                    print(f"VERIFIED: Top candidate is correctly selected (index {top_indices[0]})")



                
                best_candidates.append(prompt_best_candidates)
                all_prompt_candidates.extend(candidates)
            
            return best_candidates, all_prompt_candidates
    
    # Original strategies
    with torch.no_grad():
        if strategy == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                num_beams=generation_args["num_beams"],
                num_return_sequences=generation_args["num_return_sequences"],
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_k_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_k=generation_args["top_k"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                min_length=generation_args.get("min_length", 0),
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_p_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        else:
            raise ValueError("Unsupported strategy. Use 'beam_search', 'top_k_sampling', 'top_p_sampling', or 'best_of_n'.")

    # Decode the generated tokens for each prompt in the batch
    generated_texts = []
    all_generated_texts = []

    for i in range(0, len(outputs.sequences), generation_args["num_return_sequences"]):
        prompt_texts = []
        prompt_only = []
        for j in range(generation_args["num_return_sequences"]):
            sequence_index = i + j  # Global index for the current sequence
            output = outputs.sequences[sequence_index]
            prompt_length = input_ids.shape[-1]  # Length of the input prompt
            new_tokens = output[prompt_length:]  # Get only the generated tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = output[:prompt_length]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
            prompt_texts.append(generated_text)
            prompt_only.append(prompt_text)

        generated_texts.append(prompt_texts)
        all_generated_texts.extend(prompt_only)
    
    return generated_texts, all_generated_texts

def process_diplomacy_for_explanations(test_data, prompt_type="full", use_game_context = True):
    """
    Process diplomacy data to create explanation prompts.
    
    Args:
        test_data: List of diplomacy game data
        prompt_type: Either "single" for one order per prompt or "full" for all orders
        
    Returns:
        Dictionary of processed prompts with appropriate keys
    """
    processed_prompts = {}
    overall_index = 0
     # format the game context prompt
    game_context = """
        IMPORTANT: This is a snapshot of an ongoing Diplomacy game. The board state shows each country's current units:
        - Units prefixed with "A" are Armies located on land territories
        - Units prefixed with "F" are Fleets located on sea spaces or coastal territories
        - Each unit belongs ONLY to the country listed before the colon (e.g., all units under "AUSTRIA:" belong to Austria)
        - Supply centers are critical territories that allow powers to build new units
        - Home supply centers (like Vienna, Budapest, and Trieste for Austria) are especially important to protect

        The "Recommended Order" is a specific move being suggested for your power (Austria). An order like "A ARM S A RUM - SEV" means "Army in Armenia supports Army in Rumania's attack on Sevastopol."

        "Potential Orders for other powers" shows what other countries might do this turn. Consider how these moves could interact with or counter your recommended order.
        """
    
    for message_idx, message in enumerate(test_data):
        # Skip messages that don't need processing
        if (message['recipient'] == 'ALL' or 
            message['recipient'] == message['sender'] or 
            message["phase"].endswith('A') or 
            message["phase"].endswith('R')):
            continue
        
        # Format board state
        formatted_board = ""
        for country, units in message['units'].items():
            formatted_board += f"{country}: {', '.join(units)}\n"
        formatted_board = formatted_board.strip()
        
        # Format opponent orders
        opponent_orders = ""
        for country, units in message['predicted_orders'][message['recipient']].items():
            opponent_orders += f"{country}: {', '.join(units)}\n"
        formatted_opponent_orders = opponent_orders.strip()
        
        # Format message history (keeping this part from original function)
        message_history = ""
        start_index = max(0, len(message['prev_5_message']) - 5)
        for msg_info in message['prev_5_message'][start_index:]:
            sender = msg_info['sender']
            text = msg_info['message']
            message_history += f"Message from {sender}: {text}\n"
        message_history += f"Message from {message['sender']}: {message['message']}"
        
        # Get recommended orders
        recommended_orders = message['predicted_orders'][message['sender']][message['recipient']]
        
        if prompt_type == "full":
            # Create a single prompt for all orders
            formatted_recommended_orders = ",\n".join(recommended_orders)
            
            # Create prompt for full set of orders
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant specializing in the Diplomacy board game. Your role is to analyze:
1. The current board state.
2. The recommended orders for the player.
3. The potential orders for every power.

Your goal is to provide a comprehensive explanation of the recommended strategy:

<belief_state>
Outline the key strategic considerations and assumptions that underlie these recommended orders, including relevant board positions, territorial control, and the overall strategic context.
</belief_state>

<rationale>
Explain the strategic logic behind these recommended orders, how they work together to advance the player's position, and why they're optimal compared to alternatives given the current board state and potential moves by other powers.
</rationale>

<friction>
Summarize the most crucial insights about this strategy that the player should understand, highlighting the key moves and their importance to the overall plan.
</friction>

Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each.
<|eot_id|><|start_header_id|>user<|end_header_id|>
**Board State:**  
{formatted_board}

**Recommended Orders for {message['recipient']}:**  
{formatted_recommended_orders}

**Potential Orders for other powers:**
{formatted_opponent_orders}

**Request:**  
You are advising the player controlling {message['recipient']}. Explain the strategic rationale behind the recommended orders.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            # Store the prompt with a unique key
            processed_prompts[f'Prompt_level2_game_{message_idx}'] = prompt
            
        elif prompt_type == "single":
            # Create individual prompts for each order
            for order_idx, order in enumerate(recommended_orders):
                # Get the country name from the raw data
                country_name = message['recipient']
                if use_game_context:
                    print("using game context", use_game_context)
                
                # Create generic game context with the country name inserted
                game_context = f"""
                IMPORTANT: This is a snapshot of an ongoing Diplomacy game. The board state shows each country's current units:
                - Units prefixed with "A" are Armies located on land territories
                - Units prefixed with "F" are Fleets located on sea spaces or coastal territories
                - Each unit belongs ONLY to the country listed before the colon (e.g., all units under "{country_name}:" belong to {country_name})
                - Supply centers are critical territories that allow powers to build new units
                - Home supply centers are especially important to protect

                The "Recommended Order" is a specific move being suggested for your power ({country_name}). 
                "Potential Orders for other powers" shows what other countries might do this turn. Consider how these moves could interact with or counter your recommended order.
                IMPORTANT DIPLOMACY ORDER SYNTAX:
                * "A VIE - GAL" means the Army in Vienna moves to Galicia
                * "F BLA S A RUM - SEV" means the Fleet in Black Sea stays in place and supports Rumania's attack on Sevastopol
                * "A BUD H" means the Army in Budapest holds in place
                * "F BLA - RUM" means the Fleet in Black Sea moves to Rumania
                * "A VEN S A ROM - VEN" means the Army in Venice supports Rome's attack on Venice (which would prevent an enemy unit from successfully moving to Venice)
                """
                
                # Create enhanced tag definitions that reference the specific country
                enhanced_tag_definitions = f"""
                <belief_state>
                {country_name} STRATEGIC CONTEXT: Analyze the current board position specifically from {country_name}'s perspective. Identify which units belong to {country_name} (listed under '{country_name}:' in the board state), what territorial objectives are relevant, and how this specific order fits into {country_name}'s broader strategy and current diplomatic situation.
                </belief_state>

                <rationale>
                ORDER-SPECIFIC ANALYSIS: Provide a thorough tactical explanation of why the specific order shown under "Recommended Order for {country_name}" makes strategic sense. Explain what this order accomplishes, how it counters threats from other powers' potential orders, and why it's optimal compared to alternatives {country_name} could make with this unit. Reference the board state and other powers' potential orders to justify your explanation.
                </rationale>

                <friction>
                KEY INSIGHT: Provide the single most important strategic insight about this order that {country_name} must understand. Explain its significance to {country_name}'s overall position and how it relates to longer-term goals or threats that {country_name} faces on the board.
                </friction>
                """
                
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an expert assistant specializing in the Diplomacy board game. Your role is to analyze:
                1. The current board state.
                2. A specific recommended order for the player.
                3. The potential orders for every power.

                {game_context if use_game_context else ""}

                Your goal is to provide a detailed explanation for this specific recommended order:

                {enhanced_tag_definitions}

                Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each. First provide the <rationale>, then <belief_state>, and finally <friction>.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                **Board State:**  
                {formatted_board}

                **Recommended Order for {country_name}:**  
                {order}

                **Potential Orders for other powers:**
                {formatted_opponent_orders}

                **Request:**  
                You are advising the player controlling {country_name}. Explain the strategic rationale behind this specific recommended order.
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
                # Store the prompt with a unique key that won't repeat
                # Split by lines, strip each line, then rejoin
                prompt = "\n".join(line.strip() for line in prompt.split("\n"))
                processed_prompts[f'Prompt_level2_game_{message_idx}_order_{order_idx}'] = prompt.strip()
                overall_index += 1
    
    return processed_prompts

def process_models_with_explanation_prompts(models_list, test_data, prompt_type="full", generation_args=None, output_dir=None):
    """
    Process formatted Diplomacy explanation prompts with multiple models.
    
    Args:
        models_list: List of model paths to process
        test_data: List of diplomacy game data
        prompt_type: Either "single" for one order per prompt or "full" for all orders
        generation_args: Arguments for text generation
        output_dir: Directory to save results
    """
    import os
    import gc
    import torch
    import pickle
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First, create the formatted prompts based on the specified type
    processed_prompts = process_diplomacy_for_explanations(test_data, prompt_type, use_game_context = True) #using some diplomacy related context since the model gets confused otherwise!!
    print(f"Number of processed prompts: {len(processed_prompts)}")
    
    results_by_model = {}
    
    for agent_model_name in tqdm(models_list, desc="Processing Models"):
        # Format model name for results
        loading_model_name = agent_model_name
        if "/" in agent_model_name:
            parts = agent_model_name.split("/")
            if len(parts) >= 3:
                agent_model_name = parts[1] + "_" + parts[2]
            elif len(parts) == 2:
                agent_model_name = parts[0] + "_" + parts[1]
        
        print(f"\n===== Processing Model: {agent_model_name} =====\n")
        
        # Load model and tokenizer
        try:
            # Load base model
            lora_model = AutoModelForCausalLM.from_pretrained(
                "llama3_8b_instruct",
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            # Apply LoRA adapter
            lora_model = PeftModel.from_pretrained(
                lora_model,
                loading_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # Merge the model
            print("Merging LoRA adapter...")
            merged_model = lora_model.merge_and_unload()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(loading_model_name)
            tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.padding_side = "right"
            
            # Initialize results for this model
            model_results = {
                "model_name": agent_model_name,
                "prompt_type": prompt_type,
                "explanations": []
            }
            
            # Process each prompt
            for prompt_key, prompt_text in processed_prompts.items():
                print(f"Processing {prompt_key}")
                
                # Extract game and order info from prompt key
                prompt_info = {
                    "prompt_key": prompt_key
                }
                
                # Parse the prompt key to extract metadata
                # Format is: Prompt_level2_game_{game_idx}_order_{order_idx} for single
                # Or: Prompt_level2_game_{game_idx} for full
                if "_order_" in prompt_key:
                    # Single order prompt
                    parts = prompt_key.split('_')
                    game_idx = int(parts[parts.index("game") + 1])
                    order_idx = int(parts[parts.index("order") + 1])
                    prompt_info["game_idx"] = game_idx
                    prompt_info["order_idx"] = order_idx
                    prompt_info["type"] = "single"
                else:
                    # Full strategy prompt
                    parts = prompt_key.split('_')
                    game_idx = int(parts[parts.index("game") + 1])
                    prompt_info["game_idx"] = game_idx
                    prompt_info["type"] = "full"
                
                # Generate responses
                generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                    merged_model, 
                    tokenizer, 
                    prompt_text, 
                    generation_args, 
                    None,
                    strategy="top_p_sampling", 
                    batched=True, 
                )
                
                # Process the generated text
                if generated_texts and isinstance(generated_texts, list):
                    text_to_parse = generated_texts[0][0] if (generated_texts[0] and isinstance(generated_texts[0], list)) else generated_texts[0]
                    
                    # Parse components
                    tags_for_parsing = ["friction", "rationale", "belief_state"]
                    parsed_components = parse_tags_robust(text_to_parse, tags_for_parsing)
                    
                    # Extract components
                    friction_component = ' '.join(parsed_components.get('friction', []))
                    if not friction_component:
                        friction_component = handle_friction_logic(text_to_parse)
                    
                    belief_state = ' '.join(parsed_components.get('belief_state', []))
                    rationale = ' '.join(parsed_components.get('rationale', []))
                    
                    # Save results for this prompt
                    explanation_result = {
                        **prompt_info,  # Include all the parsed metadata
                        "raw_prompt": prompt_text,
                        "generated_text": text_to_parse,
                        "parsed_components": {
                            "friction": friction_component,  # Now used for key insights/summary
                            "rationale": rationale,          # Strategic logic/explanation
                            "belief_state": belief_state     # Strategic context/assumptions
                        }
                    }
                    
                    model_results["explanations"].append(explanation_result)
            
            # Save results for this model to a pickle file
            safe_model_name = agent_model_name.replace('/', '_')
            with open(f"{output_dir}/explanation_results_{prompt_type}_{safe_model_name}.pkl", 'wb') as f:
                pickle.dump(model_results, f)
            
            # Also store in our overall results dictionary
            results_by_model[agent_model_name] = model_results
            
            # Clean up the model to free memory
            del merged_model, tokenizer, lora_model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing model {agent_model_name}: {str(e)}")
            continue
    
    return results_by_model

if __name__ == "__main__":
    #load json raw data for diplomacy prompt construction
    test_path = f"diplomacy_prompts/sample_1K_denis_testdata_with_prompts.json"
    with open(test_path, "r") as file:
        test_data = json.load(file)

    # create the test prompts in friction generation format with xml tags for specific fields
    
    # test_prompts = process_diplomacy_for_friction(test_data[0:500]) # select the first 500 prompts to get more model samples at first. 

    #CHANGE THE NAMES OF THESE TO WHAT YOUR WEIGHT FOLDER NAMES ARE!!
    models_list_deli_weights =[
        

  
 
        'sft_deli_multiturn_rogue_cleaned/checkpoint-3000',
 
 
     'DELI_all_weights/DELI_dpo_weights/checkpoint-3500',
 
    # 'DELI_all_weights/DELI_ipo_weights/checkpoint-4500',

        "DELI_all_weights/DELI_sft_weights/checkpoint-small-1500",
         "DELI_all_weights/DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800",
            'DELI_all_weights/DELI_faaf_weights/checkpoint-2000',
 
    ]
    #sample generation arguments -- for sampling, tokens to generate etc
    generation_args = {
    "max_new_tokens": 356,
    "temperature": 0.9,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "num_beams": 5,
    "min_length": 100,
    'num_return_sequences': 1,
    "sampling_strategy": "top_p_sampling"
    }

    output_dir = "diplomacy_prompts/init_generations_single_explanation"
    #Note: we are only using 500 samples for this run, hence test_data[0:500] — each such sample contains three types of prompts
    # results_by_model = process_models_with_formatted_prompts(models_list_deli_weights,test_data[0:500], generation_args, output_dir = output_dir) #this was for the original run where friction was emphasized
    process_models_with_explanation_prompts(models_list_deli_weights, test_data[0:200], prompt_type="single", generation_args=generation_args, output_dir=output_dir)
    # process_models_with_explanation_prompts(models_list_deli_weights, test_data, prompt_type="full", generation_args=generation_args, output_dir=output_dir)
    