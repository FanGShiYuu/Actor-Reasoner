## Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicles Interactions

### [Shiyu Fang](https://fangshiyuu.github.io/)

## Abstract
Autonomous Vehicles (AVs) have entered the commercialization stage, but their limited ability to interact and express intentions still poses challenges in interactions with Human-driven Vehicles (HVs). Recent advances in large language models (LLMs) enable bidirectional human-machine communication, but the conflict between slow inference speed and the need for real-time decision-making challenges practical deployment.
To address these issues, this paper introduces a parallel Actor-Reasoner framework designed to enable explicit bidirectional AV-HV interactions across multiple scenarios. First, by facilitating interactions between the LLM-driven Reasoner and heterogeneous simulated HVs during training, an interaction memory database, referred to as the Actor, is established. Then, by introducing the memory partition module and the two-layer memory retrieval module, the Actor's ability to handle heterogeneous HVs is significantly enhanced. Ablation studies and comparisons with other decision-making methods demonstrate that the proposed Actor-Reasoner framework significantly improves safety and efficiency. Finally, with the combination of the eHMI display information derived from Reasoner’s reasoning and the feasible action solutions retrieved from the Actor, the effectiveness of the proposed Actor-Reasoner is confirmed in multi-scenario field interactions.

## Methodology Overview
Complex and dynamic driving interactions place high demands on the real-time performance and interpretability of AV decision-making. Additionally, the heterogeneity of agents and the diversity of scenarios further amplify these challenges. To address these issues, this paper proposes an LLM-driven Actor-Reasoner framework, which combines fast system retrieval with slow system reasoning to enable effective driving interactions tailored to heterogeneous HVs and diverse scenarios.

<img src="./src/tr-framework.png" alt="framework" width="1000"/>

The above figure depicts the overall framework of the proposed Actor-Reasoner architecture. This architecture draws on two cognitive modes from behavioral science theory when thinking: intuitive fast reactions and deliberate slow reasoning \citep{christakopoulou2024agentsthinkingfastslow, kahneman2011thinking}. That is, in real-world decision-making, the human brain alternates between instinctively generating quick actions based on experience and formulating well-thought-out strategies after careful consideration. Building on this concept, a parallel framework comprising an LLM-based fast system and a Two-layer Memory Retrieval-based (TMemR-based) slow system is designed. This architecture enables the identification of heterogeneous driver styles, the design of eHMI display information, and the rapid generation of AV decisions during driving interactions.

In addition to the Actor-Reasoner architecture, the proposed framework incorporates a heterogeneous driver reproduction module and an environment module. During the training phase, a non-cooperative Bayesian game is utilized to model the decision-making processes of heterogeneous HVs, which are subsequently simulated for interaction within a virtual simulative environment. In the testing phase, experiments are carried out in a real-world test field, where HVs are manually operated by experienced drivers.

## Experiments and Analysis

This section primarily serves as a supplement to the original paper’s experiment. First, we visualize the field test cases presented in the paper. In the experiment, human drivers communicate their intentions verbally through headsets. This information is converted into text and transmitted to the autonomous vehicle (AV) via the On-Board Unit (OBU). Upon receiving the human instruction, the Actor-Reasoner generates both the eHMI information and the AV's driving decisions based on the test protocol. This system outputs CAV behavior (at 10 Hz) and eHMI information accordingly.

<img src="./src/tr-field case.png" alt="tr-field case" width="1000"/>

We further visualized this interaction process through video as shown below.

<div style="text-align: center;">
  <video muted controls width="500" style="margin-right: 10px;">
    <source src="./src/case.mp4" type="video/mp4">
  </video>
</div>

In the video above, the CAV is driven by our designed Actor-Reasoner. In this scenario, the HV entered the intersection earlier and had a higher initial speed. Consequently, the Reasoner classified the HV as an aggressive driver, prompting the eHMI to display “I will be Slower” to convey the AV’s intention. Based on this assessment, the Actor retrieved a deceleration action to allow the HV to pass through the intersection first.  

<source src="./src/case.mp4" type="video/mp4">

However, the HV driver did not follow the AV’s guidance and instead chose to stop at 15s, verbally stating, “I will be slower.” After converting this speech input to text and feeding it into both the Reasoner and the Actor, the Reasoner first interpreted the HV’s intent and, after reasoning, decided to display "I will be Faster" on the eHMI to ensure efficient interaction. The Actor then generated an acceleration decision in real-time, ultimately allowing the AV to pass through the intersection first.

### Field test at different scenarios

Due to space constraints in the paper, we were unable to present detailed experimental cases. Here, we have selected several representative interaction cases from different scenarios to further analyze the feasibility of applying the proposed method in various contexts.

First, we present the actual operation of the AV in a roundabout scenario, where the human driver exhibits different intentions. In the left image, after the AV receives and understands the HV's intention to go first, it chooses to slow down and yield, ensuring a safe interaction. In a scenario with the same initial positions, the driver decides to go last and slows down, prompting the AV to go first, thus improving the efficiency of the interaction.

| <video muted controls width=380> <source src="./src/roundabout_av_rush.mp4"  type="video/mp4"> </video> <video muted controls width=380> <source src="./src/roundabout_av_yield.mp4"  type="video/mp4"> </video> |

Additionally, we present two sets of interaction examples in merging environments. Due to terrain constraints, we selected a left-turn and straight-through merge scenario at a T-junction for demonstration. Based on the standardized scenario descriptions involved, the proposed method effectively handles interactions in such scenarios and generates different behaviors depending on the specific scenario and experiential context.

| <video muted controls width=380> <source src="./src/merge_av_rush.mp4"  type="video/mp4"> </video> <video muted controls width=380> <source src="./src/merge_av_yield.mp4"  type="video/mp4"> </video> |

