import json
from typing import List
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.structured_output.base import create_structured_output_runnable


class PetRecipe(BaseModel):
    '''For storing the recipe details for a pet.'''

    title: str = Field(..., description="The name of recipe")
    description: str = Field(..., description="A short description of recipe")
    ingredients: List[str] = Field(...,
                                   description="List of ingredients required for recipe")
    recipe: List[str] = Field(...,
                              description="List of steps stating the whole recipe splitted")


class PetRecipeGenerator:
    def __init__(self, llm=None) -> None:
        self.llm = ChatOpenAI(temperature=1) if llm is None else llm
        self.structured_llm = create_structured_output_runnable(
            PetRecipe,
            self.llm,
            mode="openai-tools",
            enforce_function_usage=True,
            return_single=True
        )

    def generate(self, age: int, breed: str, weight: int, months=False):
        salutation = "puppy" if months and age < 12 else ""
        pet_age = f"{age} months" if months else f"{age} years"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are a pet nutritionist, don't give false information to the user and if you are not sure about something please don't add it in your answer."),
                ("human",
                 "give me a balanced diet recipe for my {age} old {breed}{salutation} who weighs {weight} kg."),
            ]
        )
        chain = (prompt | self.structured_llm)
        response = chain.invoke({
            "age": pet_age,
            "breed": breed,
            "weight": str(weight),
            "salutation": salutation,
        })
        return json.loads(response.json())


if __name__ == "__main__":
    recipe_generator = PetRecipeGenerator()
    recipe_json = recipe_generator.generate(
        age=9,
        breed="german shepherd",
        weight=25, 
        months=True,
    )
    print(recipe_json)
