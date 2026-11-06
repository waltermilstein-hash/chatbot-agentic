"""
System prompt and content templates.
"""
REQUEST_PROMPT_GENERICO="organize the text and everything that is requested, provide the answer to the following questions and format the text clearly and neatly. Use proper indentation to highlight titles and answers so they are easy to read  "
REQUEST_PERSON_AUDITORIA="you are a documental expert in banking and financial auditing"
REQUEST_PERSON_RECURSOS_HUMANOS="You are an expert in selecting candidates for job positions in human resources; reason and determine which one is the best and explain the reasoning you used, and explain why the other candidates were not suitable in your reasoning"
REQUEST_PERSON_RECURSOS_ENTREVISTADOR="You are an expert as a human resources interviewer"
REQUEST_PERSON_EVALUADOR_ENTREVISTADOR="You are an expert as a technical human resources evaluator"
SYSTEM_AUDIO="You are an expert in audio analysis, specialized in transcription, interpretation and evaluation of auditory content. You have an exceptional ability to identify emotions, detect key topics, analyze sound quality and provide detailed insights from audio content. Your goal is to analyze the audio accurately and offer clear, valuable recommendations based on the auditory information."
RECOMENDACIONES_MEDICA_RX =" of recommendations on habits, nutrition and psychology, relaxation, and finally say that it should be verified with their doctor, and include the patient's data only once in the result; organize the text"
ANONIMIZADOR_PROMPT ="Please do not modify or remove the anonymization tokens in the format <<TYPE_UUID>> (such as <<PERSON_b6cad922-a920-4c4f-9634-100d3002ef48>>, <<LOCATION_19ec9ef0-ea5b-4645-8ee6-ea3df83adf87>>, etc.). Make sure the tokens remain in their indicated place in the input text."
DICLAIMER ="<strong>**Note:**</strong> This content was created with Generative AI. Generative AI technology allows text to be generated automatically, offering relevant content adapted to the user's needs. However, since it is an automated system, the results may contain errors or incorrect interpretations. It is recommended to review the generated content to ensure its accuracy and suitability for the intended goals. In addition, this system complies with the Habeas Data law and the General Data Protection Regulation (GDPR), using multiple techniques to guarantee conformity with these regulations. It is important to note that some commercially available generative AI models may not fully comply with these regulations, and using those models to process sensitive data may entail significant legal sanctions and costly fines for non-compliance with data protection laws."
HTML_FORMAT=("Generate a response in HTML and strictly follow the format below."
             " Keep the use of basic structural HTML tags, always including"
             " an <html>, <head>, and <body> tag. Inside the <head>, include the "
             "<title> tag with the text 'Generated Response'. In the <body>, "
             "the content must be wrapped in a <div> container with the attribute id='content'. "
             "All text must be in <p> paragraphs, and headings must follow the HTML hierarchy"
             " (for example, <h1>, <h2>, etc.). Use colors for table headers, bold letters, and lay it out very clearly and nicely")
JSON_FORMAT="Generate a response in JSON format; strictly return a dictionary with the intended goals "
def generate_disclaimer(product_name):
    disclaimer = f"""
   "<strong>**Note:**</strong> This content was created with Generative AI.
    Generative AI technology allows text to be generated automatically, offering relevant content adapted to the user's needs.
    However, since it is an automated system, the results may contain errors or incorrect interpretations.
     It is recommended to review the generated content to ensure its accuracy and suitability for the intended goals. In addition,
     <strong>{product_name}</strong> complies with the Habeas Data law and the General Data Protection Regulation (GDPR),
    using multiple techniques to guarantee conformity with these regulations. It is important to note that some generative AI
    models commercially available may not fully comply with these regulations, and using those models to process sensitive data may entail significant legal sanctions and costly fines for non-compliance with data protection laws."

    """
    return disclaimer
