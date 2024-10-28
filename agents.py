from crewai import Agent, Task, Crew, Process
import os
from datetime import datetime
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import HumanMessage
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
search_tool = DuckDuckGoSearchRun()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=3000
)

class MarketResearchCrew:
    def __init__(self):
        # Initialize agents
        self.industry_researcher = Agent(
            role='Industry Research Specialist',
            goal='Research industry trends and company information',
            backstory="""You are an experienced industry analyst with expertise in identifying market trends.""",
            tools=[search_tool],
            verbose=True
        )

        self.use_case_generator = Agent(
            role='AI Solutions Architect',
            goal='Generate AI use cases based on industry research',
            backstory="""You specialize in identifying opportunities for AI implementation.""",
            tools=[search_tool],
            verbose=True
        )

        self.resource_collector = Agent(
            role='Technical Resource Specialist',
            goal='Find relevant datasets and resources',
            backstory="""You excel at finding datasets and resources for AI implementations.""",
            tools=[search_tool],
            verbose=True
        )

    def research_company(self, company_name: str) -> str:
    # Create tasks for the crew
        research_task = Task(
            description=f"""Research {company_name} and their industry thoroughly.
            1. Identify the main industry sector
            2. Analyze key products/services
            3. Identify strategic focus areas
            4. Research current technology adoption
            
            Provide a comprehensive report in the following format:
            - Industry Overview
            - Company Position
            - Current Technology Stack
            - Key Challenges
            """,
            agent=self.industry_researcher,
            expected_output="""{"Industry_Overview":"","Company_Position":"","Technology_Analysis":"","Market_Trends":"","Competitive_Landscape":""}"""
        )

        use_case_task = Task(
            description=f"""Based on the industry research for {company_name}, generate AI use cases.
            1. Identify areas where AI can improve operations
            2. Suggest GenAI implementations for:
            - Customer service
            - Internal operations
            - Product development
            3. Prioritize use cases by:
            - Implementation feasibility
            - Expected impact
            - Resource requirements
            
            Present findings in a structured format with clear justification for each use case.""",
            agent=self.use_case_generator,
            context=[research_task],
            expected_output="""{"Detailed_description":"","Expected_business_benefits":"","Implementation_complexity":"","Required_resources_and_capabilities":"","Potential_challenges_and_risks":"","ROI_indicators":""}"""
        )

        resource_task = Task(
            description=f"""For the identified use cases, collect relevant resources:
            1. Find datasets on Kaggle/HuggingFace/GitHub
            2. Identify relevant research papers
            3. List similar implementations
            4. Suggest technical frameworks
            
            Organize resources by use case and include direct links.""",
            agent=self.resource_collector,
            context=[research_task,use_case_task],
            expected_output="""{"Provide_direct_links":"","Brief_description":"","Relevance_to_use_case":"","Usage_considerations":""}"""
        )

        # Create the crew and start the process
        crew = Crew(
            agents=[self.industry_researcher, self.use_case_generator, self.resource_collector],
            tasks=[research_task, use_case_task, resource_task],
            process=Process.sequential
        )

        # Start the crew process
        results = crew.kickoff()
        # Aggregate results
        report = self.aggregate_results(results)
        
        # Generate a comprehensive report using the LLM
        response = llm.invoke([HumanMessage(content=f"Generate a comprehensive report based on the following results:\n{report}")])
        
        print("Result:",response)
        # Return the generated report
        return response.content

    def aggregate_results(self, results) -> str:
        """Aggregate results into a formatted string."""
        combined_results = ""
        for result in results:
            combined_results += f"{result}\n\n"
        return combined_results

def save_results(company_name: str, results: str) -> str:
    """Save results to a file and return the filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"market_research_{company_name.lower().replace(' ', '')}_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(results)
    
    return filename

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Market Research & Use Case Generation System')
    parser.add_argument('--company', '-c', help='Company name to analyze')
    args = parser.parse_args()

    # Initialize the crew
    market_research_crew = MarketResearchCrew()

    # Get company name either from command line argument or user input
    company_name = args.company
    if not company_name:
        company_name = input("Enter the company name to analyze: ")

    # Perform market research and generate a report
    result = market_research_crew.research_company(company_name)
    filename = save_results(company_name, result)
    print(f"\nAnalysis complete! Results saved to {filename}")

if __name__ == "__main__":
    main()