"""
CV Link Checker - Complete Standalone Version
===============================================
මේ program එකෙන් CV PDF එකක තියෙන LinkedIn සහ GitHub links 
හොයාගෙන ඒවා වැඩ කරනවද කියලා check කරනවා.

Usage:
    python check_my_cv.py "path/to/your/cv.pdf"
    
or

    python check_my_cv.py  (එතකොට CV path එක අහනවා)
"""

import re
import requests
import sys
from urllib.parse import urlparse

try:
    import pymupdf4llm
except ImportError:
    print("❌ Error: pymupdf4llm library එක install වෙලා නැහැ!")
    print("\n📦 Install කරන්න මේ command එක run කරන්න:")
    print("   pip install pymupdf4llm requests --break-system-packages")
    sys.exit(1)


def extract_all_links(cv_text):
    """CV text එකෙන් සියලුම web links extract කරනවා"""
    
    # LinkedIn links
    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/[^\s\)\]<>"\']+|linkedin\.com/[^\s\)\]<>"\']+'
    linkedin_links = re.findall(linkedin_pattern, cv_text, re.IGNORECASE)
    
    # GitHub links
    github_pattern = r'https?://(?:www\.)?github\.com/[^\s\)\]<>"\']+|github\.com/[^\s\)\]<>"\']+'
    github_links = re.findall(github_pattern, cv_text, re.IGNORECASE)
    
    # Portfolio/Personal website links
    portfolio_pattern = r'https?://[^\s\)\]<>"\']+\.github\.io[^\s\)\]<>"\']*'
    portfolio_links = re.findall(portfolio_pattern, cv_text)
    
    return linkedin_links, github_links, portfolio_links


def check_link_validity(url):
    """Link එක accessible වෙනවද කියලා check කරනවා"""
    
    if not url.startswith('http'):
        url = 'https://' + url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        
        if 200 <= response.status_code < 400:
            return True, response.status_code, "✓ වැඩ කරනවා"
        elif response.status_code == 404:
            return False, response.status_code, "✗ 404 - හම්බ වුණේ නැහැ"
        elif response.status_code == 403:
            return False, response.status_code, "⚠ 403 - Access denied"
        else:
            return False, response.status_code, f"✗ Status {response.status_code}"
    
    except requests.exceptions.Timeout:
        return False, None, "⏱ Timeout"
    except requests.exceptions.SSLError:
        return False, None, "🔒 SSL Error"
    except requests.exceptions.ConnectionError:
        return False, None, "🔌 Connection Error"
    except Exception as e:
        return False, None, f"❌ Error: {str(e)[:50]}"


def analyze_cv(cv_text):
    """CV එකේ සියලුම links විස්තරාත්මකව analyze කරනවා"""
    
    print("\n" + "=" * 70)
    print(" " * 20 + "📄 CV LINK ANALYZER 📄")
    print("=" * 70)
    
    linkedin_links, github_links, portfolio_links = extract_all_links(cv_text)
    
    results = {
        'linkedin': {'found': len(linkedin_links), 'working': 0, 'broken': 0},
        'github': {'found': len(github_links), 'working': 0, 'broken': 0},
        'portfolio': {'found': len(portfolio_links), 'working': 0, 'broken': 0}
    }
    
    # LinkedIn Links
    print("\n🔵 LINKEDIN LINKS:")
    print("-" * 70)
    if linkedin_links:
        for i, link in enumerate(linkedin_links, 1):
            print(f"\n  [{i}] {link}")
            is_valid, status_code, message = check_link_validity(link)
            print(f"      {message}", end="")
            if status_code:
                print(f" (Status: {status_code})")
            else:
                print()
            
            if is_valid:
                results['linkedin']['working'] += 1
            else:
                results['linkedin']['broken'] += 1
    else:
        print("  ⚠ LinkedIn link එකක් හම්බ වුණේ නැහැ!")
        print("  💡 Tip: LinkedIn profile link එකක් add කරන්න!")
    
    # GitHub Links
    print("\n" + "=" * 70)
    print("🟢 GITHUB LINKS:")
    print("-" * 70)
    if github_links:
        for i, link in enumerate(github_links, 1):
            print(f"\n  [{i}] {link}")
            is_valid, status_code, message = check_link_validity(link)
            print(f"      {message}", end="")
            if status_code:
                print(f" (Status: {status_code})")
            else:
                print()
            
            if is_valid:
                results['github']['working'] += 1
            else:
                results['github']['broken'] += 1
    else:
        print("  ⚠ GitHub link එකක් හම්බ වුණේ නැහැ!")
        print("  💡 Tip: GitHub profile link එකක් add කරන්න!")
    
    # Portfolio Links
    print("\n" + "=" * 70)
    print("🌐 PORTFOLIO/WEBSITE LINKS:")
    print("-" * 70)
    if portfolio_links:
        for i, link in enumerate(portfolio_links, 1):
            print(f"\n  [{i}] {link}")
            is_valid, status_code, message = check_link_validity(link)
            print(f"      {message}", end="")
            if status_code:
                print(f" (Status: {status_code})")
            else:
                print()
            
            if is_valid:
                results['portfolio']['working'] += 1
            else:
                results['portfolio']['broken'] += 1
    else:
        print("  ℹ Portfolio link එකක් හම්බ වුණේ නැහැ")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 සාරාංශය (SUMMARY):")
    print("=" * 70)
    print(f"\n  LinkedIn Links:   හම්බුණා: {results['linkedin']['found']:2d} | "
          f"වැඩ කරනවා: {results['linkedin']['working']:2d} | "
          f"වැඩ කරන්නේ නැහැ: {results['linkedin']['broken']:2d}")
    
    print(f"  GitHub Links:     හම්බුණා: {results['github']['found']:2d} | "
          f"වැඩ කරනවා: {results['github']['working']:2d} | "
          f"වැඩ කරන්නේ නැහැ: {results['github']['broken']:2d}")
    
    print(f"  Portfolio Links:  හම්බුණා: {results['portfolio']['found']:2d} | "
          f"වැඩ කරනවා: {results['portfolio']['working']:2d} | "
          f"වැඩ කරන්නේ නැහැ: {results['portfolio']['broken']:2d}")
    
    # Overall Score
    total_found = results['linkedin']['found'] + results['github']['found']
    total_working = results['linkedin']['working'] + results['github']['working']
    
    print("\n" + "-" * 70)
    if total_found == 0:
        print("  ⚠️  WARNING: LinkedIn සහ GitHub links නැහැ!")
        print("  💡 TIP: Professional links add කරන්න CV එකට!")
    elif total_working == total_found and total_found > 0:
        print("  ✅ EXCELLENT: සියලුම links වැඩ කරනවා!")
    elif total_working > 0:
        print(f"  ⚠️  ATTENTION: {total_found - total_working} link(s) වැඩ කරන්නේ නැහැ!")
        print("  💡 TIP: වැඩ නොකරන links fix කරන්න!")
    else:
        print("  ❌ CRITICAL: කිසිම link එකක් වැඩ කරන්නේ නැහැ!")
    
    print("=" * 70 + "\n")
    
    return results


def main():
    """Main function - CV path එක input ගෙන analyze කරනවා"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "🎓 CV Link Checker - Standalone 🎓")
    print("=" * 70)
    
    # CV path එක get කරනවා
    if len(sys.argv) > 1:
        cv_path = sys.argv[1]
    else:
        print("\n📁 CV PDF file එකේ path එක enter කරන්න:")
        print("   Example: C:\\Users\\YourName\\Downloads\\Your_CV.pdf")
        print("   හෝ: /home/user/documents/cv.pdf")
        cv_path = input("\n➡️  Path: ").strip().strip('"').strip("'")
    
    # Path එක validate කරනවා
    if not cv_path:
        print("\n❌ Error: Path එකක් දීලා නැහැ!")
        return
    
    print(f"\n📄 CV එක load කරනවා: {cv_path}")
    
    try:
        # PDF එකෙන් text extract කරනවා
        print("⏳ Processing... (මේකට කිහිපයක් තත්පර යයි)")
        cv_text = pymupdf4llm.to_markdown(cv_path)
        
        print("✅ CV එක load වුණා!")
        print("🔍 Links check කරනවා... (Internet connection එකක් ඕන)")
        print("-" * 70)
        
        # Analysis එක run කරනවා
        results = analyze_cv(cv_text)
        
        print("\n✅ Analysis එක complete වුණා!")
        print("📧 ප්‍රශ්න තිබ්බොත් හෝ suggestions තිබ්බොත් කියන්න!")
        
    except FileNotFoundError:
        print(f"\n❌ Error: CV file එක හම්බ වුණේ නැහැ!")
        print(f"   Path: {cv_path}")
        print("\n💡 Tips:")
        print("   - Path එක හරියට ඇතුළත් කරන්න")
        print("   - File එක එ location එකේ තියෙනවද කියලා check කරන්න")
        print("   - Windows path නම් quotes use කරන්න: \"C:\\path\\to\\file.pdf\"")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\n💡 Common issues:")
        print("   - PDF එක corrupt වෙලා නැද්ද බලන්න")
        print("   - PDF එක encrypted/password protected නැද්ද බලන්න")
        print("   - File එක ඔයා use කරනවද? (ඔයා open කරලා තියෙනවද?)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program එක cancel වුණා!")
        print("👋 Bye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")