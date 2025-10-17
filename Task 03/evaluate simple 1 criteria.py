import pymupdf4llm
import re
import time


def does_the_github_links_mentioned(text):
    github_pattern = r'https?://github\.com/[\w\-]+/?[\w\-]*'

    # සියලු GitHub links හොයාගන්නවා
    links = re.findall(github_pattern, text)

    if links:
        print(f"✓ GitHub links තියෙනවා! ({len(links)} links හම්බුණා)\n")
        print("හම්බුණු Links:")
        for i, link in enumerate(links, 1):
            print(f"{i}. {link}")
        return True
    else:
        print("✗ GitHub links නැහැ!")
        return False




markdown = pymupdf4llm.to_markdown(r"C:\Users\Vikasitha\Documents\RP\Sample CVs\K.K.H.DEWMINI_RESUME (3).pdf")
print(markdown)

print(does_the_github_links_mentioned(markdown))


