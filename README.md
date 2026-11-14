# HackPrinceton-Fall-2026

Project Description: Fall detection using a camera and microphone that is integrated into a raspberry pi that would call first responders if a person is unresponsive

Deep Agent that transitions from stages if it observes an anomaly:
MONITORING: This layer is the standby layer where it collects frames from the camera and audio signal from the room. It stays at this state when it hasn't detected an abnormal behavior such as collapsing to the ground by a human via YOLO and (a soon to be created voice-detection system).
EVALUATING: Once the keywords such as "help" or the camera sees someone collapsed on the ground, the Deep Agent evaluates the situation with various subagents. If those subagents activate and agree something bad has happened, the Deep Agent (using LangChain's Deep Agent LLM) sends that up to the Computer Vision Model (utilized by Gemini and connected via Gemini API). Gemini review the computer vision model and makes an analysis of the image with a short description of the event after being given the prompt.
CHALLENGING: If the Deep Agent receives a description suggesting that the person has indeed collapsed and/or may need help, the Deep Agent tells speaker to repeat a few phrases to ask if the person needs help.
AWAITING_RESPONSE: This is the final confirmation stage where a subagent responsible for the audio interpretation runs in the background awaiting for keywords that either suggest the person is fine, or does indeed need help. Should the person not be fine, the model will then move into escalation mode.
ESCALATION: Send an SMS message to a trusted contact (potentially emergency services). Also sends the images it reviews to a frontend (this is for the purposes of ensuring that the images that were analyzed indeed do suggest a potential safety situation).

Group members Format:
Name (Discord user)

Group members
Joshua (trixler_329);
Victoria (uv3p3);
Sujin (s_shk_);
Martin (theothermartintm)