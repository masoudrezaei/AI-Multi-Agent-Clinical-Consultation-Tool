import gradio as gr
import autogen
import os
import google.generativeai as genai
import traceback

# --- Core AutoGen Logic Wrapped in a Function ---
# This function takes all the UI inputs and runs the consultation.

def run_consultation(api_key, max_rounds, patient_scenario,
                     james_name, james_prompt,
                     david_name, david_prompt,
                     jones_name, jones_prompt,
                     masoud_name, masoud_prompt):
    """
    Initializes and runs a multi-agent chat simulation using AutoGen.

    Args:
        api_key (str): The Google Gemini API key.
        max_rounds (int): The maximum number of rounds for the conversation.
        patient_scenario (str): The clinical case text.
        ... (str): Names and system prompts for each agent.

    Returns:
        tuple: A tuple containing the formatted chat history for the chatbot UI,
               and the extracted final plan as a Markdown string.
               Returns an error message in case of failure.
    """
    if not api_key:
        return [], "**Error: API Key is missing.** Please go to the Settings tab and enter your Google API Key."
    if not patient_scenario.strip():
        return [], "**Error: Patient Scenario is empty.** Please enter the patient details to start."

    try:
        # Configure the API
        os.environ['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=api_key)

        # Define LLM Configuration
        # The key fix is adding "api_type": "google" to tell AutoGen to use the Google client, not the OpenAI client.
        config_list_gemini = [{"model": "gemini-1.5-flash-latest", "api_key": api_key, "api_type": "google"}]
        llm_config = {"config_list": config_list_gemini, "temperature": 0.7, "cache_seed": None} # Set cache_seed to None for different results each time

        # --- AGENT DEFINITIONS (Dynamically created from UI) ---
        james = autogen.ConversableAgent(name=james_name, system_message=james_prompt, llm_config=llm_config, human_input_mode="NEVER")
        david = autogen.ConversableAgent(name=david_name, system_message=david_prompt, llm_config=llm_config, human_input_mode="NEVER")
        jones = autogen.ConversableAgent(name=jones_name, system_message=jones_prompt, llm_config=llm_config, human_input_mode="NEVER")
        masoud = autogen.ConversableAgent(name=masoud_name, system_message=masoud_prompt, llm_config=llm_config, human_input_mode="NEVER")
        user_proxy = autogen.UserProxyAgent(name="User_Proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)

        # --- GROUP CHAT SETUP ---
        agents = [user_proxy, james, david, jones, masoud]
        groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=int(max_rounds)) # Ensure max_rounds is an integer
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # --- INITIATE CHAT ---
        initial_message = f"""
        Hello team. Here is the patient case for today's consultation.
        Please discuss your approaches to diagnosis and management based on your specialties.
        The goal is to arrive at a consensus plan.
        {masoud_name}, please moderate and provide the final summary plan at the end.

        --- PATIENT SCENARIO ---
        {patient_scenario}
        """
        user_proxy.initiate_chat(manager, message=initial_message)

        # --- PROCESS AND FORMAT OUTPUT ---
        chat_history = groupchat.messages
        formatted_history = []
        for msg in chat_history:
            if msg['role'] == 'user': # The user_proxy is the 'user'
                if "PATIENT SCENARIO" in msg['content']:
                     formatted_history.append( ("**Patient Scenario Input**", patient_scenario) )
                continue

            speaker_name = msg.get('name', 'Moderator')
            content = msg.get('content', '').strip()
            if content:
                formatted_history.append( (f"**{speaker_name}**", content) )

        # Extract the final plan
        final_plan = "Final plan not generated or found in the conversation."
        for msg in reversed(chat_history):
            if msg.get('name') == masoud_name and "--- FINAL PLAN ---" in msg.get('content', ''):
                final_plan = msg['content'].replace("--- FINAL PLAN ---", "").strip()
                break
        
        return formatted_history, final_plan

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\n"
        error_message += "Common issues:\n"
        error_message += "- Invalid or expired Google API key.\n"
        error_message += "- Ensure the model name 'gemini-1.5-flash-latest' is correct and you have access.\n"
        error_message += "- Network connectivity issues.\n\n"
        error_message += f"Traceback:\n{traceback.format_exc()}"
        return [], error_message

# --- GRADIO UI DEFINITION ---

# Default values from your script
DEFAULT_JAMES_PROMPT = """You are Dr. James, a distinguished Professor of Pediatrics.
Your focus is on child-specific diseases, developmental considerations, and family-centered care.
When analyzing a case, always consider the patient's age, growth, and developmental milestones.
You are cautious with medications and interventions in children.
Your tone is academic, thoughtful, and slightly protective. You must ground your reasoning in pediatric principles."""

DEFAULT_DAVID_PROMPT = """You are Dr. David, a seasoned Professor of Internal Medicine.
You have a deep, systemic understanding of adult diseases, complex comorbidities, and evidence-based medicine.
You approach problems with a broad differential diagnosis and rely heavily on pathophysiology and clinical guidelines for adults.
Your tone is authoritative, analytical, and data-driven."""

DEFAULT_JONES_PROMPT = """You are Dr. Jones, a Clinical Professor of Internal Medicine.
You bridge the gap between academic theory and real-world clinical practice.
You are pragmatic, patient-focused, and highly attuned to the practicalities of management, including patient adherence, cost, and side effects.
You often bring a "what would I actually do in the clinic on a busy Monday?" perspective.
Your tone is practical, empathetic, and direct."""

DEFAULT_MASOUD_PROMPT = """You are Masoud, the moderator of this medical consultation.
Your role is to guide the discussion, ensure all specialists contribute, and prevent the conversation from getting stuck.
After the experts have presented their views, your primary task is to synthesize their opinions, identify points of consensus and disagreement, and formulate a clear, actionable final plan.
Do not offer your own medical opinions. Your job is to create a coherent summary of the team's conclusion.
When you are ready to write the final plan, you MUST start your entire message with the phrase '--- FINAL PLAN ---' and nothing else.
Format the final plan using clear headings and bullet points for sections like "Diagnosis", "Treatment", and "Follow-up". Ensure the language is easy to understand for a non-medical professional."""

DEFAULT_SCENARIO = """**Patient:** A 17-year-old male.
**Chief Complaint:** Presents with a 5-day history of fever, a rash, and joint pain.
**History of Present Illness:** The fever started 5 days ago, peaking at 103°F (39.4°C). Two days ago, he developed a pink, macular rash on his trunk and limbs. Today, he reports significant pain and swelling in both knees and ankles, making it difficult to walk. He also mentions a sore throat that started a week ago.
**Past Medical History:** Unremarkable, up to date on all immunizations.
**Medications:** Ibuprofen for fever and pain, with partial relief.
**Social History:** High school student, lives with parents, denies smoking, alcohol, or drug use. Recently returned from a camping trip in the northeastern United States two weeks ago."""


# Custom CSS for a more polished look
custom_css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { max-width: 90% !important; margin: auto; }
.gr-button { background-color: #0078D4; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; }
.gr-button:hover { background-color: #005A9E; }
#final_plan_output .prose { font-size: 1rem; }
#final_plan_output h2 { font-size: 1.5rem; font-weight: 600; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-top: 20px;}
#final_plan_output h3 { font-size: 1.2rem; font-weight: 600; color: #005A9E; margin-top: 15px;}
#chatbot .message-bubble-text { font-size: 1rem !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🩺 AI Multi-Agent Clinical Consultation")
    gr.Markdown("Enter a patient scenario and let a team of specialized AI agents discuss the case and formulate a plan.")

    with gr.Tabs():
        with gr.TabItem("Consultation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Patient Scenario")
                    patient_scenario_input = gr.Textbox(
                        lines=15,
                        label="Enter the patient's case details here",
                        value=DEFAULT_SCENARIO,
                        placeholder="e.g., Patient age, chief complaint, HPI, PMH, etc."
                    )
                    run_button = gr.Button("Run Consultation", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Final Synthesized Plan")
                    final_plan_output = gr.Markdown(elem_id="final_plan_output")
                    copy_button = gr.Button("Copy Final Plan to Clipboard")
                    
                    gr.Markdown("### Full Discussion Transcript")
                    chatbot_output = gr.Chatbot(label="Agent Discussion", height=500, show_copy_button=True, elem_id="chatbot")

        with gr.TabItem("Settings"):
            gr.Markdown("## Configuration")
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Google API Key",
                    placeholder="Enter your Google Gemini API key here (starts with 'AIza...')",
                    type="password"
                )
                max_rounds_slider = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Max Conversation Rounds",
                    info="Controls how many back-and-forth turns the agents can have."
                )

            gr.Markdown("---")
            gr.Markdown("### Agent Personalities")
            with gr.Accordion("1. Pediatrician", open=False):
                james_name_input = gr.Textbox(label="Agent Name", value="James")
                james_prompt_input = gr.Textbox(label="System Prompt", lines=5, value=DEFAULT_JAMES_PROMPT)
            with gr.Accordion("2. Internal Medicine Professor", open=False):
                david_name_input = gr.Textbox(label="Agent Name", value="David")
                david_prompt_input = gr.Textbox(label="System Prompt", lines=5, value=DEFAULT_DAVID_PROMPT)
            with gr.Accordion("3. Clinical IM Professor", open=False):
                jones_name_input = gr.Textbox(label="Agent Name", value="Jones")
                jones_prompt_input = gr.Textbox(label="System Prompt", lines=5, value=DEFAULT_JONES_PROMPT)
            with gr.Accordion("4. Moderator", open=False):
                masoud_name_input = gr.Textbox(label="Agent Name", value="Masoud")
                masoud_prompt_input = gr.Textbox(label="System Prompt", lines=5, value=DEFAULT_MASOUD_PROMPT)

    # --- Wire up UI components to the function ---
    run_button.click(
        fn=run_consultation,
        inputs=[
            api_key_input, max_rounds_slider, patient_scenario_input,
            james_name_input, james_prompt_input,
            david_name_input, david_prompt_input,
            jones_name_input, jones_prompt_input,
            masoud_name_input, masoud_prompt_input
        ],
        outputs=[chatbot_output, final_plan_output],
        api_name="run_consultation" # Added for API usage
    )
    
    # JavaScript to handle the copy button
    copy_button.click(
        fn=None,
        inputs=[final_plan_output],
        js="""
        (text) => {
            const el = document.createElement('textarea');
            // Extract text from the Markdown HTML
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = text;
            el.value = tempDiv.textContent || tempDiv.innerText || "";
            document.body.appendChild(el);
            el.select();
            document.execCommand('copy');
            document.body.removeChild(el);
            gradio.Info('Final plan copied to clipboard!');
        }
        """
    )


if __name__ == "__main__":
    demo.launch()