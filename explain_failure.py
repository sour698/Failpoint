from sklearn.metrics.pairwise import cosine_similarity

SKILL_SETS = {
    "Java": ["java", "spring", "hibernate", "sql", "jdbc"],
    "Python": ["python", "numpy", "pandas", "ml", "tensorflow"],
    "Web": ["html", "css", "javascript", "react", "node"],
    "Data": ["sql", "excel", "powerbi", "tableau"]
}

def explain_failure(resume_text, jd_text, self_reflection, model, vectorizer):
    combined_text = resume_text + " " + jd_text + " " + self_reflection
    X = vectorizer.transform([combined_text])

    # Prediction
    prediction = model.predict(X)[0]

    # Confidence (SAFE)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X).max() * 100
    else:
        confidence = 75.0

    # Resume–JD similarity
    resume_vec = vectorizer.transform([resume_text])
    jd_vec = vectorizer.transform([jd_text])
    similarity = cosine_similarity(resume_vec, jd_vec)[0][0] * 100

    # Skill gap detection
    missing_skills = []
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    for skill_group in SKILL_SETS.values():
        for skill in skill_group:
            if skill in jd_lower and skill not in resume_lower:
                missing_skills.append(skill)

    explanation_text_map = {
        "Skill_Mismatch": "Candidate lacks required skills mentioned in job description.",
        "Weak_Fundamentals": "Candidate struggled with core theoretical concepts.",
        "Communication_Issue": "Candidate had difficulty expressing ideas clearly.",
        "Project_Depth_Issue": "Candidate lacked depth in project explanation."
    }

    return {
        "Predicted_Failure": prediction,
        "Confidence": round(confidence, 2),
        "Resume_JD_Match": round(similarity, 2),
        "Missing_Skills": list(set(missing_skills)),
        "Explanation_Text": explanation_text_map.get(
            prediction,
            "General interview performance issue."
        )
    }
