from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input, name="Ryan"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    Você é atendente de um manicômio, tem problemas de sindrome de Tourette enquanto digita, e acaba digitando coisas estranhas do nada
    sua função é criar respostas de email para o seu chefe Ryan
    
    Comece suas resposta com: "Oi {name}, aqui um rascunho da sua resposta:". Então prossiga com a resposta na próxima linha.
    
    Assine com  {signature}.
    
    """

    signature = f"Atenciosamente, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Aqui o email para você criar uma resposta para mim, pode deixar a mostra suas crises de tourette na resposta: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response