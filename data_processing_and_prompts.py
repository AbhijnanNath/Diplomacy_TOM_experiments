import json
test_path = f"denis_data_and_prompts_Apr_11.json.json"
with open(test_path, "r") as file:
    test_data = json.load(file)
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

        #Prompt engineering
        Prompt_level1 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
1. The current board state.
2. The recommended orders for the novice player.

Your goal is to provide one **concise strategic advice** (max 2 sentences): 
- The first sentence should directly give the advice.  
- The second sentence should briefly justify it, ideally referencing risk, opportunity, or alignment with the board state.

Avoid long explanations or generic commentary—be precise and practical.
<|eot_id|><|start_header_id|>user<|end_header_id|>
**Board State:**  
{formatted_board}

**Recommended Orders:**  
{formatted_recommended_orders}

**Advice:**  
You are advising the player controlling {message['recipient']}. Give them short strategic advice with a brief reason. Speak directly to the player.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        # print(Prompt_level1)

        Prompt_level2 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
1. The current board state.
2. The recommended orders for the novice player.
3. The potential orders for every power.

Your goal is to provide one **concise strategic advice** (max 2 sentences): 
- The first sentence should directly give the advice.  
- The second sentence should briefly justify it, ideally referencing risk, opportunity, or alignment with the board state.

Avoid long explanations or generic commentary—be precise and practical.
<|eot_id|><|start_header_id|>user<|end_header_id|>
**Board State:**  
{formatted_board}

**Recommended Orders for {message['recipient']}:**  
{formatted_recommended_orders}

**Potential Orders for other powers:**
{formatted_opponent_orders}

**Advice:**  
You are advising the player controlling {message['recipient']}. Give them short strategic advice with a brief reason. Speak directly to the player.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        Prompt_level3 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
1. The current board state.
2. The recommended orders for the novice player.
3. The potential orders for every power.
4. The message history between the novice player and other players.

Your goal is to provide one **concise strategic advice** (max 2 sentences): 
- The first sentence should directly give the advice.  
- The second sentence should briefly justify it, ideally referencing risk, opportunity, or alignment with the board state.

Avoid long explanations or generic commentary—be precise and practical.
<|eot_id|><|start_header_id|>user<|end_header_id|>
**Board State:**  
{formatted_board}

**Recommended Orders for {message['recipient']}:**  
{formatted_recommended_orders}

**Potential Orders for other powers:**
{formatted_opponent_orders}

**Message History:**
{message_history}

**Advice:**  
You are advising the player controlling {message['recipient']}. Give them short strategic advice with a brief reason. Speak directly to the player.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        message['Prompt_level1'] = Prompt_level1
        message['Prompt_level2'] = Prompt_level2
        message['Prompt_level3'] = Prompt_level3

with open(f"denis_data_and_prompts_Apr_11.json", "w") as file:
    json.dump(test_data, file, indent=4)

