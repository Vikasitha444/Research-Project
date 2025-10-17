import pymupdf4llm
import re
import requests
import time



def extract_the_github_links(text):

    #GitHub repository links විතරක්  Extract කරනවා

    repo_pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
    repos = re.findall(repo_pattern, text)
    return list(set(repos))



def check_the_repository_exists(repo_url):
    """
    GitHub repository එක තියෙනවද කියලා check කරන්න
    """
    try:
        response = requests.head(repo_url, timeout=5, allow_redirects=True)

        if response.status_code == 200:
            return True, "✓ Repository තියෙනවා"
        elif response.status_code == 404:
            return False, "✗ Repository එක හම්බෙන්නේ නැහැ (404)"
        else:
            return False, f"✗ Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "✗ Timeout - Server respond කරේ නැහැ"
    except requests.exceptions.RequestException as e:
        return False, f"✗ Connection Error: {str(e)}"


def validate_github_repo(text):
    """
    මෙතනින් බලනවා Valid github links කියක් තිබ්බද? Invalid Links කියක් තිබ්බද කියලා
    """
    repos = extract_the_github_links(text)

    if not repos:
        print("GitHub Repository links හම්බෙන්නේ නැහැ!")
        return

    print(f"හම්බුණු Repositories: {len(repos)}\n")
    print("="*60)

    valid_count = 0
    invalid_count = 0

    for i, repo in enumerate(repos, 1):
        print(f"\n{i}. {repo}")
        exists, message = check_the_repository_exists(repo)
        print(f"   {message}")

        if exists:
            valid_count += 1
        else:
            invalid_count += 1

        time.sleep(0.5)  # Rate limiting වලට respect කරන්න

    print("\n" + "="*60)
    print(f"\nසාරාංශය:")
    print(f"  Valid Repos: {valid_count}")
    print(f"  Invalid Repos: {invalid_count}")
    print(f"  Total: {len(repos)}")


# PDF එක read කරන්න
markdown = pymupdf4llm.to_markdown(r"C:\Users\Vikasitha\Documents\RP\Sample CVs\Fake github links for practise.pdf")

print("=== GitHub Repository Validation ===\n")
validate_github_repo(markdown)