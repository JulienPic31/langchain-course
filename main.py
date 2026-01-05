from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """The Montreal Canadiens[note 4] (French: Canadiens de Montréal, lit. 'Canadians of Montreal'), officially Club de hockey Canadien (lit. 'Canadian hockey club')[9] and colloquially known as the Habs,[note 5] are a professional ice hockey team based in Montreal. The Canadiens compete in the National Hockey League (NHL) as a member of the Atlantic Division in the Eastern Conference. Since 1996, the team has played its home games at the Bell Centre, originally known as the Molson Centre.[10] The Canadiens previously played at the Montreal Forum, which housed the team for seven decades and all but their first two Stanley Cup championships.[note 6]

Founded in 1909, the Canadiens are the oldest continuously operating professional ice hockey team worldwide, and the only existing NHL club to predate the founding of the league. One of the earliest North American professional sports franchises, the Canadiens' history predates that of every other Canadian franchise outside the Canadian Football League's Toronto Argonauts, as well as every American franchise outside baseball and the National Football League's Arizona Cardinals. The franchise is one of the "Original Six", the teams that made up the NHL from 1942 until the 1967 expansion. The team's championship season in 1992–93 marked the last time a Canadian team won the Stanley Cup.[11][12]

The Canadiens have won the Stanley Cup 24 times, more times than any other franchise, having earned 23 victories since the founding of the NHL, and 22 since 1927, when NHL teams became the only ones to compete for the Stanley Cup.[13] The Canadiens also had the most championships by a team of any of the major North American sports leagues until the New York Yankees won their 25th World Series title in 1999.[14]
"""
    summary_template = """
    given the information {information} about a thing I want you to create:
    - a short summary of it
    - two interesting facts about it
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:1b")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
