import gradio as gr


def launch_chat_interface(chat_func):
    """
    Launch the Gradio chat interface with dark theme enabled.

    This sets up a conversational UI using the provided chat function, enforcing
    a dark mode by modifying the URL query parameters via JavaScript.

    Args:
        chat_func (callable): The function to handle chat input and return responses.
            Should accept (question: str, history: list) and return str.
    """
    force_dark_mode = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    gr.ChatInterface(chat_func, type="messages", js=force_dark_mode).launch(
        inbrowser=True,
    )
