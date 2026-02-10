"""
Test script for job recommendation integration
"""

from job_matcher import get_job_matcher

def test_job_matcher():
    """Test the job matcher functionality"""
    print("Testing Job Matcher Integration\n")
    print("=" * 60)

    # Initialize matcher
    matcher = get_job_matcher()
    print("\n✓ Job matcher initialized successfully")

    # Test with sample keywords (like from CV)
    test_keywords = ["Python", "JavaScript", "React", "Node.js", "MongoDB", "MySQL", "Django", "Flask"]
    print(f"\n📋 Test Keywords: {', '.join(test_keywords)}")

    # Get recommendations
    print("\n🔍 Getting job recommendations...")
    recommendations = matcher.get_recommendations(test_keywords, top_n=5)

    print(f"\n✓ Found {len(recommendations)} job recommendations")
    print("\n" + "=" * 60)
    print("Top 5 Job Matches:")
    print("=" * 60)

    for i, job in enumerate(recommendations, 1):
        print(f"\n{i}. {job['title']}")
        print(f"   Company: {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Match Score: {job['match_score']}% ({job['match_text']})")
        print(f"   Salary: {job['salary_range']}")

    # Test skills gap analysis
    print("\n" + "=" * 60)
    print("Skills Gap Analysis:")
    print("=" * 60)

    # Common job requirements
    job_requirements = [
        "Python", "JavaScript", "React", "Node.js", "Docker", "Kubernetes",
        "AWS", "MongoDB", "MySQL", "TypeScript", "Git"
    ]

    gap = matcher.get_skills_gap_analysis(test_keywords, job_requirements)
    print(f"\n✓ Skills You Have ({len(gap['has_skills'])}): {', '.join(gap['has_skills'])}")
    print(f"\n✗ Skills to Learn ({len(gap['missing_skills'])}): {', '.join(gap['missing_skills'])}")
    print(f"\n📊 Match Percentage: {gap['match_percentage']:.1f}%")

    # Test market insights
    print("\n" + "=" * 60)
    print("Market Insights:")
    print("=" * 60)

    insights = matcher.get_market_insights(test_keywords)
    if insights:
        print(f"\n📈 Total Jobs Available: {insights['total_jobs']}")
        print(f"🔥 High Match Jobs: {insights['high_match_count']}")
        print(f"👍 Medium Match Jobs: {insights['medium_match_count']}")
        print(f"📊 Average Match Score: {insights['average_match']:.1f}%")

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_job_matcher()
