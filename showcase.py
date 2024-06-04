from transformers import pipeline

# Choose your desired UI framework (Tkinter for simplicity here)
import tkinter as tk


def answer_question(question, context):
    """
    Answers a question based on the provided context using a QA pipeline.
    """
    pipe = pipeline("question-answering", model="botcon/LUKE_squad_finetuned_qa_tf32")
    result = pipe(question=question, context=context)
    return result["answer"]


def get_entities(context):
    """
    Uses spaCy to identify and return entities from the context.
    You can replace spaCy with other NLP libraries for entity recognition.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")  # Load a spaCy model for English
    doc = nlp(context)
    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    return entities


def ask_and_answer():
    """
    Retrieves question and context, calls answer_question, and displays answer.
    """
    question = question_entry.get()
    context = context_text.get("1.0", tk.END)
    answer = answer_question(question, context)
    answer_label.config(text=f"Answer: {answer}")

    # Get and display entities in a separate window
    entities = get_entities(context)
    if entities:
        entity_window.title("Recognized Entities")
        entity_text.delete("1.0", tk.END)
        entity_text.insert("1.0", "\n".join(entities))
        entity_window.deiconify()  # Show the entity window
    else:
        entity_window.withdraw()  # Hide the entity window if no entities found


# Create the main window with some styling
window = tk.Tk()
window.title("QA Demonstration about TU/Eindhoven")
window.geometry("800x600")  # Set window size

# Define labels and entry fields for question and context with styling
question_label = tk.Label(window, text="Enter your question:", font=("Arial", 12))
question_label.pack(pady=10)

question_entry = tk.Entry(window, width=70, font=("Arial", 14))
question_entry.pack(pady=10)

context_label = tk.Label(window, text="Context:", font=("Arial", 12))
context_label.pack(pady=10)

context_text = tk.Text(window, width=70, height=15, font=("Arial", 12))
context_text.insert("1.0", """The Eindhoven University of Technology (TU/e), established in 1956, is a prominent public technical university in the Netherlands. It's known for its focus on collaboration between industry, government, and academia.  TU/e offers a wide range of Bachelor's and Master's programs in various engineering fields, with a growing emphasis on computer science and data science. The university boasts a vibrant research environment and is a significant contributor to technological advancements.""")
context_text.pack(pady=10)

answer_label = tk.Label(window, text="", font=("Arial", 12))
answer_label.pack(pady=10)


# Create a button to trigger question answering
ask_button = tk.Button(window, text="Ask a question", command=ask_and_answer, font=("Arial", 12))
ask_button.pack(pady=10)

# Create a separate window to display entities (unchanged)
entity_window = tk.Tk()
entity_window.withdraw()  # Initially hide the entity window

entity_text = tk.Text(entity_window)
entity_text.pack()


# Run the main event loop
window.mainloop()
