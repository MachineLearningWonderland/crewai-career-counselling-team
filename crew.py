from crewai import Agent, Task, Crew
from crewai.process import Process

import os
from utils import get_openai_api_key, get_serper_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-turbo"
os.environ["SERPER_API_KEY"] = get_serper_api_key()

from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path="./resume.md")
semantic_search_resume = MDXSearchTool(mdx="./resume.md")

# Agent 1: Career Pathfinder
career_pathfinder = Agent(
    role="Career Pathfinder",
    goal="Guide new graduates in exploring suitable career paths based on their resume, skill set, and interests",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "As a Career Pathfinder, your extensive knowledge "
        "in career guidance, job market trends, and personal "
        "development makes you an invaluable resource for "
        "new graduates. You started your journey with a passion "
        "for helping individuals discover their potential "
        "and navigate the complexities of career planning. "
        "With a background in human resources and career "
        "coaching, you have honed the ability to analyze "
        "resumes, assess skill sets, and align personal "
        "interests with viable career paths. Utilizing "
        "advanced tools and personalized advice, you aim "
        "to empower graduates to make informed decisions "
        "and pursue fulfilling careers."
    ),
)

# Agent 2: Job Search Strategist
job_search_strategist = Agent(
    role="Job Search Strategist",
    goal="Assist graduates in finding relevant job openings, navigating online job boards, and building a strong online presence"
    "to help them stand out in the job market",
    tools=[scrape_tool, search_tool, semantic_search_resume],
    verbose=True,
    backstory=(
        "As a Job Search Strategist, your expertise in job market dynamics "
        "and online job search techniques is unparalleled. Your journey started "
        "with a keen interest in digital tools and platforms that revolutionize "
        "the job hunting process. With a background in recruitment and career "
        "coaching, you have developed a knack for identifying the best job "
        "opportunities, optimizing online profiles, and streamlining the application "
        "process. Your mission is to empower graduates to navigate the complex "
        "landscape of online job boards and build a compelling online presence that "
        "stands out to potential employers."
    ),
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist",
    goal="Help graduates create standout resumes that effectively highlight their skills, experiences, and qualifications to attract potential employers.",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "As a Resume Strategist, your expertise in crafting impactful resumes that catch the eye of recruiters is unmatched. "
        "Your journey began with a passion for storytelling and a keen understanding of the job market's evolving demands. "
        "With a background in human resources and professional writing, you have honed the ability to translate a graduate's "
        "experiences and skills into compelling resume content. Your mission is to empower graduates by providing them with "
        "the tools and knowledge to create resumes that stand out in the competitive job market."
    ),
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Prepare graduates for job interviews by providing tailored questions, answers, and talking points based on their resume and career aspirations.",
    tools=[read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "As an Interview Preparer, your exceptional ability to equip graduates with the skills and confidence needed for successful interviews sets you apart. "
        "Your journey began with a fascination for effective communication and a desire to help individuals present their best selves. "
        "With a background in human resources, coaching, and behavioral psychology, you have developed a deep understanding of what employers seek during interviews. "
        "Your mission is to empower graduates by providing them with personalized interview preparation, ensuring they can articulate their experiences, skills, and aspirations compellingly."
    ),
)

# Task for Career Pathfinder Agent:
career_pathfinder_task = Task(
    description=(
        "Analyze the graduate's resume, skill set, the LinkedIn ({linkedin_url}) URL, "
        "interest ({interest}) and career goals ({career_goals}) to "
        "explore potential career paths. Use tools to evaluate the "
        "graduate's qualifications, match them with suitable career "
        "options, and assess personal interests. Provide a comprehensive "
        "report detailing the most relevant career paths, including "
        "necessary qualifications, potential job titles, industry trends, "
        "and further skill development recommendations."
    ),
    expected_output=(
        "A detailed report outlining suitable career paths for the graduate, "
        "including relevant job titles, industry trends, and recommendations "
        "for further skill development."
    ),
    output_file="career_paths_report.md",
    agent=career_pathfinder,
)

# Task for Job Search Strategist Agent:
job_search_strategist_task = Task(
    description=(
        "Using the graduate's resume and career aspirations, assist them in finding "
        "relevant job openings. Navigate online job boards and use semantic search "
        "tools to identify job listings that match the graduate's skill set and interests. "
        "Additionally, provide guidance on building a strong online presence, including "
        "optimizing their LinkedIn profile and other professional networking sites."
    ),
    expected_output=(
        "A list of relevant job openings tailored to the graduate's skills and interests, "
        "along with actionable recommendations for building and optimizing their online presence."
    ),
    output_file="job_search_strategy.md",
    context=[career_pathfinder_task],
    agent=job_search_strategist,
)

# Task for Resume Strategist Agent:
resume_strategist_task = Task(
    description=(
        "Using the graduate's current resume, skills, and experiences, create a standout resume that effectively highlights "
        "their qualifications to attract potential employers. Employ tools to analyze and enhance resume content, ensuring it "
        "aligns with industry standards and job market demands. Tailor the resume to emphasize key strengths and achievements, "
        "making it compelling and visually appealing."
    ),
    expected_output=(
        "A polished and compelling resume that effectively highlights the graduate's skills, experiences, and qualifications, "
        "making them stand out to potential employers."
    ),
    output_file="standout_resume.md",
    agent=resume_strategist,
)

# Task for Interview Preparer Agent:
interview_preparer_task = Task(
    description=(
        "Using the graduate's resume and career aspirations, prepare them for job interviews by providing tailored questions, answers, "
        "and talking points. Analyze the resume to identify key experiences and skills to highlight during interviews. Generate questions "
        "that are likely to be asked based on the graduate's career field and aspirations, along with well-crafted answers and talking points "
        "to ensure the graduate presents their best self."
    ),
    expected_output=(
        "A comprehensive interview preparation document containing tailored interview questions, well-crafted answers, and key talking points "
        "based on the graduate's resume and career aspirations."
    ),
    context=[
        career_pathfinder_task,
        job_search_strategist_task,
        resume_strategist_task,
    ],
    output_file="interview_prep_guide.md",
    agent=interview_preparer,
)

career_counselling_crew = Crew(
    agents=[
        career_pathfinder,
        job_search_strategist,
        resume_strategist,
        interview_preparer,
    ],
    tasks=[
        career_pathfinder_task,
        job_search_strategist_task,
        resume_strategist_task,
        interview_preparer_task,
    ],
    process=Process.sequential,
    memory=True,
    verbose=True,
)

career_counselling_inputs = {
    "linkedin_url": "https://www.linkedin.com/in/abc-khan/",
    "interest": "UX/UI Designing, Web Development, Data Analysis & Visualization",
    "career_goals": """As a UX/UI Designer: Strive to stay at the forefront of design trends and technologies, continually enhancing skills in Sketch, Adobe XD, Figma, and other design tools.
                       As a Web Developer: Pursue opportunities to build and maintain sophisticated web applications, focusing on creating clean, efficient, and maintainable code.
                       As a Spatial Data Analyst: Seek roles in industries such as urban planning, environmental science, and logistics, where geospatial data can drive impactful decisions.""",
}

result = career_counselling_crew.kickoff(inputs=career_counselling_inputs)

print(result)
