from github import Github
from dotenv import load_dotenv
import os
load_dotenv()
g = Github(os.getenv("GITHUB_TOKEN"))
print(g.get_user().get_repo("macc-project"))