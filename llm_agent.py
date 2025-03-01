import copy
import time
from prompt_llm import *
import subprocess
import os

def run_llama3(prompt):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    process = subprocess.Popen(
        ["ollama", "run", "llama3"],  # llama3, mistral, deepseek-r1, gemma
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    output, _ = process.communicate(input=prompt)
    return output

class LLM_Agent():
    def __init__(self):
        self.pre_prompt = PRE_DEF_PROMPT()
        self.toolModels = [
            getAvailableActions(),
            isAccelerationConflictWithCar(),
            isKeepSpeedConflictWithCar(),
            isDecelerationSafe(),
        ]

    def llm_run(self, decision_info, instruction_info, ego_info, other_info, memory_tools, if_train_mode=True):
        current_scenario = self.prompt_engineer(ego_info, other_info)  # prompt engineer
        llm_output = self.send_to_llm(decision_info, instruction_info, current_scenario, memory_tools, if_train_mode)
        if if_train_mode:
            self.add_memory2dataset(llm_output, ego_info, other_info, instruction_info, memory_tools)
        return llm_output

    def relative_memory(self, memory, prompt_info):
        experience = ""
        extract_prompt = prompt_info.strip().split('\n')
        query_scenario = '\n'.join(extract_prompt)
        past_decisions = memory.retrieveMemory(query_scenario, top_k=2)[0]
        for past_decision in past_decisions:
            experience += f"- Last time {past_decision['negotiation_result']}, you choose to {past_decision['final_action']}, it is {past_decision['comments']}\n"
        experience += f"Above messages are some examples of how you make a decision in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario."
        return experience

    def send_to_llm(self, decision_info, instruction_info, current_scenario, memory_tools, if_train_mode):
        # past_memory = self.relative_memory(memory_tools, prompt_info)
        past_memory = ''
        if if_train_mode:
            prompt = (f"{self.pre_prompt.SYSTEM_MESSAGE_PREFIX}"
                      f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
                      # "There are several rules you need to follow when you drive:"
                      # f"{self.pre_prompt.get_traffic_rules()}\n\n"
                      "Here are your attention points:"
                      f"{self.pre_prompt.get_decision_cautions()}\n"
                      "Here is the current scenario:\n"
                      f"{current_scenario}\n"
                      # "Here is your action when scenarios are similar to the current scenario in the past, you should learn from past memory try not to take the cation that cause more danger:\n"
                      # f"{past_memory}\n\n"
                      "Your decision and shared intention to other vehicles at the last moment are as follows, try not to change them frequently unless it will cause danger:"
                      f"Your last decision is {decision_info[0]}, your shared intention to surrounding vehicle is {decision_info[1]} \n"
                      "Here is a message from the driver of surrounding vehicle. You should decide whether to accept his advice based on the current scenario, and express your intention to surrounding vehicles\n"
                      "They won not lie to you, and when them want to slow down or yield, you should better faster for efficiency\n"
                      f"Surrounding vehicle says: {instruction_info}\n"
                      "You can choose your decision and your intention"
                      "You need to consider the following questions step by step to reach your conclusion:\n"
                      "1. What is the likely intention of the surrounding vehicle considering its past actions? (e.g., ACCELERATE or DECELERATE?)\n"
                      "2. Based on its past actions and your estimation on its intention, what is its driving style? (e.g., AGGRESSIVE or CONSERVATIVE)\n"
                      "3. What action should you take next based on surrounding vehicle's intention and style? Your action better not consistent with surrounding vehicle action. e.g. If surrounding vehicle tend to slow down, you better FASTER for efficiency (e.g., IDLE, FASTER, SLOWER)\n"
                      "4. What you going to do next, what is your intention? (e.g., I will faster/slower/stop/go first/...)\n"
                      "Most importantly, your action must align with your intention"
                      "After reasoning, provide your final conclusion in the following format:\n"
                      "```\n"
                      "ONLY OUTPUT YOUR FINAL DECISION IN THE FOLLOWING FORMAT AND NOTHING ELSE:\n"
                      "```\n"
                      "Final Answer: \n"
                      "    \"thoughts\": {\"<your thoughts when consider the above questions step by step to reach your conclusion)>\"},\n"
                      "    \"surrounding vehicle intention\": {\"<surrounding vehicle's intention, only output ACCELERATE or DECELERATE)>\"},\n"
                      "    \"style\": {\"<surrounding surrounding's driving style, only output AGGRESSIVE or CONSERVATIVE)>\"},\n"
                      "    \"decision\": {\"<ego car's decision, ONE of the available actions which is IDLE, FASTER, SLOWER)>\"},\n"
                      "    \"your intention to share\": {\"<What you going to do like: I will XXX (less than five word, easy to understand, output with lowercase)>\"} \n"
                      "```\n")
        else:  # without output thought, faster inference
            prompt = (f"{self.pre_prompt.SYSTEM_MESSAGE_PREFIX}"
                      f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
                      "Here are your attention points:"
                      f"{self.pre_prompt.get_decision_cautions()}\n"
                      "Here is the current scenario:\n"
                      f"{current_scenario}\n"
                      "Your decision and shared intention to other vehicles at the last moment are as follows, try not to change them frequently unless it will cause danger:"
                      f"Your last decision is {decision_info[0]}, your shared intention to surrounding vehicle is {decision_info[1]} \n"
                      f"At last frame, you estimation on surrounding vehicle's driving style is {decision_info[3]}:"
                      "Here is a message from the driver of surrounding vehicle. You should decide whether to accept his advice based on the current scenario, and express your intention to surrounding vehicles\n"
                      f"Surrounding vehicle says: {instruction_info}\n"
                      "You need to consider the following questions step by step to reach your conclusion:\n"
                      "1. What is the likely intention of the surrounding vehicle considering its past actions? (e.g., ACCELERATE or DECELERATE?)\n"
                      "2. Based on its past actions and your estimation on its intention, what is its driving style? (e.g., AGGRESSIVE or CONSERVATIVE)\n"
                      "3. What action should you take next, if you think surrounding vehicle is going to DECELERATE, you'd better FASTER. (e.g., IDLE, FASTER, SLOWER)\n"
                      "4. Shared your intention (What you going to do) to surrounding vehicle based on your action. (e.g., I will ....)\n"
                      "ONLY OUTPUT YOUR FINAL DECISION IN THE FOLLOWING FORMAT AND NOTHING ELSE (DO NOT OUTPUT YOUR THOUGHTS):\n"
                      "```\n"
                      "Final Answer: \n"
                      "    \"surrounding vehicle intention\": {\"<surrounding vehicle's intention, only output ACCELERATE or DECELERATE)>\"},\n"
                      "    \"style\": {\"<surrounding surrounding's driving style, only output AGGRESSIVE or CONSERVATIVE)>\"},\n"
                      "    \"decision\": {\"<ego car's decision, ONE of the available actions which is IDLE, FASTER, SLOWER)>\"},\n"
                      "    \"your intention to share\": {\"<What you going to do and reasons (LESS THAN EIGHT WORDS)>\"} \n"
                      "```\n")
        time1 = time.time()
        # print('--------------------------------------------------------------------------------------')
        # print(prompt)
        llm_response = run_llama3(prompt)
        # print('inference time/s:', time.time() - time1)
        # print('output:', llm_response)
        llm_action, hmi_advice, inter_intention, inter_style = self.extract_output(llm_response)
        return [llm_action, hmi_advice, inter_intention, inter_style]

    def extract_output(self, response_content):
        try:
            start = response_content.find('"decision": {') + len('"decision": {')
            end = response_content.find('}', start)
            decision = response_content[start:end].strip().strip('"')

            start_advice = response_content.find('"your intention to share": {') + len('"your intention to share": {')
            end_advice = response_content.find('}', start_advice)
            advice = response_content[start_advice:end_advice].strip().strip('"')

            start_intention = response_content.find('"surrounding vehicle intention": {') + len('"surrounding vehicle intention": {')
            end_intention = response_content.find('}', start_intention)
            intention = response_content[start_intention:end_intention].strip().strip('"')

            start_style = response_content.find('"style": {') + len('"style": {')
            end_style = response_content.find('}', start_style)
            style = response_content[start_style:end_style].strip().strip('"')

            if "FASTER" in decision:
                decision = "FASTER"
            elif "SLOWER" in decision:
                decision = "SLOWER"
            elif "IDLE" in decision:
                decision = "IDLE"

            llm_output = [decision, advice, intention, style]
            for i, output in enumerate(llm_output):
                if len(output) > 100:
                    if i == 0:
                        llm_output[i] = 'SLOWER'
                    elif i == 3:
                        llm_output[i] = 'GENERAL'
                    else:
                        llm_output[i] = None
            decision, advice, intention, style = llm_output
            return decision, advice, intention, style
        except Exception as e:
            print(f"Error when extract decision: {e}")
            return None, None, None, None

    def prompt_engineer(self, ego_info, other_info):
        msg0 = available_action(self.toolModels, ego_info)
        msg1 = interaction_vehicle(ego_info, other_info)
        safety_msg = check_safety_in_current_lane(self.toolModels, ego_info, other_info)
        safety_msg2 = check_safety_with_conflict_vehicles(ego_info, other_info)
        safety_msg3 = check_conflict_point_occupied(ego_info, other_info)
        prompt_info = format_decision_info(msg0, msg1, safety_msg, safety_msg2, safety_msg3)  # msg0, msg2, availabel_lane, safety_assessment, safety_msg
        return prompt_info

    def add_memory2dataset(self, llm_output, ego_info, other_info, instruction_info, memory_tools):
        if not tools.if_passed_conflict_point(ego_info, other_info):
            action = llm_output[0]
            if action not in ['FASTER', 'SLOWER', 'IDLE']:
                pass
            else:
                sce_descrip = tools.scenario_experience_generator(ego_info, other_info, llm_output, instruction_info)
                current_delta_ttcp = tools.get_delta_ttcp(ego_info, other_info)
                ego_info_next = copy.deepcopy(ego_info)
                if action == 'FASTER':
                    ego_acc = 2
                elif action == 'SLOWER':
                    ego_acc = -2
                else:
                    ego_acc = 0
                ego_info_next.acc = ego_acc
                ego_info_next.speed += ego_acc * Dt
                next_delta_ttcp = tools.get_delta_ttcp(ego_info_next, other_info)
                if abs(current_delta_ttcp) > 5:  # can be replaced with other preset rule
                    memory_tools.addMemory(sce_descrip, action)
                else:
                    if abs(next_delta_ttcp) < abs(current_delta_ttcp):
                        pass
                    else:
                        memory_tools.addMemory(sce_descrip, action)
        else:
            pass





