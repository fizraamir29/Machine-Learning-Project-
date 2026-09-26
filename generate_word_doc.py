"""
generate_word_doc.py
Generates a beautifully styled Word document (.docx) containing the entire Machine Learning project guide,
file walkthroughs, ML concepts, embedded graphs, and Teacher Viva Q&A in Roman Urdu and English.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = Path(__file__).resolve().parent
DOCX_PATH = BASE_DIR / "Student_Engagement_ML_Complete_Guide.docx"
OUTPUTS_DIR = BASE_DIR / "outputs"


def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_callout(doc, title, text_en, text_ur):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15

    run_title = p.add_run(f"💡 {title}\n")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(11)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(27, 54, 93)

    run_en = p.add_run(f"Technical English: {text_en}\n\n")
    run_en.font.name = "Calibri"
    run_en.font.size = Pt(10)
    run_en.font.color.rgb = RGBColor(40, 40, 40)

    run_ur = p.add_run(f"🗣️ Teacher ko samjhane ke liye (Roman Urdu):\n\"{text_ur}\"")
    run_ur.font.name = "Calibri"
    run_ur.font.size = Pt(10)
    run_ur.font.italic = True
    run_ur.font.color.rgb = RGBColor(15, 82, 186)

    doc.add_paragraph()  # spacing


def build_document():
    doc = docx.Document()

    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # -------------------------------------------------------------
    # Cover / Header Title
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(2)
    r_title = p_title.add_run("Real-Time Student Engagement Detection\nusing Classical Machine Learning")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run("Complete Project Architecture, Code Walkthrough, ML Theory & Viva Defense Guide")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(90, 105, 120)

    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Author / Student:", "Fizra Amir (fizraamir29)"),
        ("Project Domain:", "Classical Machine Learning & Computer Vision"),
        ("GitHub Repository:", "https://github.com/fizraamir29/Machine-Learning-Project-"),
        ("Target Classes (6):", "Bored, Confused, Drowsy, Focused, Frustrated, Looking Away"),
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        c1 = meta_table.cell(row_idx, 0)
        c2 = meta_table.cell(row_idx, 1)
        c1.text = k
        c2.text = v
        c1.paragraphs[0].runs[0].font.bold = True
        set_cell_background(c1, "EAEFF5")
        set_cell_background(c2, "F7F9FC")
        set_cell_margins(c1, 60, 60, 100, 100)
        set_cell_margins(c2, 60, 60, 100, 100)

    doc.add_paragraph()

    # -------------------------------------------------------------
    # Section 1: Executive Overview & Why Classical ML
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Project Overview", level=1)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(6)

    p_intro = doc.add_paragraph(
        "Is project ka maqsad classroom video feed se students ke chehre real-time mein detect karna aur "
        "unka engagement state accurately 6 classes mein classify karna hai. Pehle yeh project Deep Learning "
        "(MobileNetV2 / CNN) par mushtamil tha, lekin ab isse 100% Classical Machine Learning mein transform "
        "kar diya gaya hai."
    )
    p_intro.paragraph_format.line_spacing = 1.15

    add_callout(
        doc,
        "Teacher ko batayein: Deep Learning kyun hataya aur Classical ML kyun use kiya?",
        "Deep Neural Networks act as black boxes where features are implicitly learned with millions of parameters, "
        "demanding heavy GPU compute and risking batch normalization collapse on smaller datasets (~2,000 images). "
        "Classical ML provides explicit, interpretable feature engineering (HOG, LBP, Color, Haar) and achieves "
        "sub-millisecond inference (~3ms) on CPU with superior test accuracy (93.08%).",
        "Ma'am, deep learning black-box hota hai aur heavy GPU mangta hai. Humne CNN ki jagah manual Feature Engineering ki hai "
        "jisme humne HOG (edges), LBP (texture), HSV (color), aur Haar-like features khud calculate kiye. Yeh lightweight hai, "
        "laptop ke normal CPU par bina GPU ke chalta hai, aur iski test accuracy 93.08% aayi hai jo ke deep learning se bhi behtar hai!"
    )

    # -------------------------------------------------------------
    # Section 2: Dataset & Preprocessing
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. Dataset & Preprocessing Architecture", level=1)
    h2.paragraph_format.space_before = Pt(16)
    h2.paragraph_format.space_after = Pt(6)

    p_data = doc.add_paragraph(
        "Project mein total 2,277 verified facial images hain jo 3 raw datasets se merge ki gayi hain:\n"
        "• Kaggle Student Engagement Dataset: 2,120 images\n"
        "• Maleha Dataset: 32 images\n"
        "• Zoha Dataset: 125 images (HEIC mobile photos auto-converted to JPG)\n\n"
        "Data Split Strategy (70 / 15 / 15 Stratified Split):\n"
        "• Train Set: 1,591 images (70%)\n"
        "• Validation Set: 339 images (15%)\n"
        "• Held-out Test Set: 347 images (15%)"
    )
    p_data.paragraph_format.line_spacing = 1.15

    add_callout(
        doc,
        "ML Concept: Stratified Split kya hota hai aur kyun zaroori hai?",
        "Stratified sampling guarantees that the relative class distribution in the overall dataset is strictly "
        "preserved across all subsets (train, val, test), preventing class bias and representation skew.",
        "Stratified split ka matlab hai ke har split (train, validation, test) mein har class ka percentage barabar rahega. "
        "Agar dataset mein 12% drowsy students hain, toh train, val, aur test sab mein exactly 12% hi drowsy students honge. "
        "Is se model biased nahi hota."
    )

    # -------------------------------------------------------------
    # Section 3: File-by-File Walkthrough & Code Logic
    # -------------------------------------------------------------
    doc.add_page_break()
    h3 = doc.add_heading("3. File-by-File Code Walkthrough & Libraries", level=1)
    h3.paragraph_format.space_before = Pt(16)
    h3.paragraph_format.space_after = Pt(6)

    # --- File 1: model.py ---
    doc.add_heading("File 1: src/model.py — Hand-Crafted Feature Engineering", level=2)
    p_f1 = doc.add_paragraph(
        "Yeh is poore project ka dil (core) hai! CNN ko replace karke yeh module har face crop se 938-dimensional "
        "feature vector extract karta hai. Har image standardize hoke 96x96 pixels ki banti hai."
    )
    p_f1.paragraph_format.line_spacing = 1.15

    # Table of Features
    f_table = doc.add_table(rows=5, cols=4)
    f_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Feature Type", "Dimensions", "Library Used", "Kya Capture Karta Hai?"]
    for i, h in enumerate(headers):
        cell = f_table.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_background(cell, "1B365D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    feat_rows = [
        ("HOG (Histogram of Oriented Gradients)", "800 dims", "skimage.feature.hog", "Facial structure, contours, jawline, eye angles"),
        ("Spatial LBP (Local Binary Patterns)", "90 dims", "skimage.feature.local_binary_pattern", "Skin micro-texture, wrinkles, eye open/closed status"),
        ("HSV Color Histograms", "32 dims", "cv2 (OpenCV)", "Lighting-invariant skin tone (Hue: 16, Sat: 8, Val: 8)"),
        ("Haar-like Landmark Contrasts", "16 dims", "NumPy", "Eyes vs Forehead, Mouth vs Nose, Left/Right symmetry"),
    ]
    for r_idx, r_data in enumerate(feat_rows):
        for c_idx, val in enumerate(r_data):
            cell = f_table.cell(r_idx + 1, c_idx)
            cell.text = val
            set_cell_background(cell, "F7F9FC" if r_idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, 50, 50, 80, 80)

    doc.add_paragraph()
    add_callout(
        doc,
        "Teacher ko batayein: 938 features kaise calculate huay?",
        "HOG uses 8 orientation bins x 2x2 cells/block x 5x5 blocks = 800 features. "
        "Spatial LBP splits the face into a 3x3 grid, computing 10 uniform histogram bins per cell = 90 features. "
        "HSV histograms allocate 16 (H) + 8 (S) + 8 (V) = 32 features. "
        "Haar differences compute 16 facial region contrasts. Sum: 800 + 90 + 32 + 16 = 938.",
        "Ma'am, 96x96 image se: HOG ke 800 features facial shape batate hain, LBP ke 90 features skin texture aur wrinkles "
        "batate hain, HSV ke 32 features skin color lighting se alag karte hain, aur 16 Haar features aankh aur maathay ke "
        "contrast differences batate hain. Total 938 features bante hain."
    )

    # --- File 2: train.py ---
    doc.add_heading("File 2: src/train.py & src/train_ml.py — Model Training & Cross-Validation", level=2)
    p_f2 = doc.add_paragraph(
        "Yeh script saare training data par features extract karti hai, StandardScaler se normalize karti hai, "
        "aur 3 classical ML models train aur tune karti hai:\n"
        "1. SVM (Support Vector Machine): SVC with RBF kernel, C=10.0 (Champion Model)\n"
        "2. Random Forest: 200 Decision Trees ka ensemble\n"
        "3. K-Nearest Neighbors (KNN): k=7, inverse distance-weighted Euclidean voting\n"
        "4. 5-Fold StratifiedKFold: Hyperparameter tuning comparison table generate karti hai."
    )
    p_f2.paragraph_format.line_spacing = 1.15

    add_callout(
        doc,
        "Teacher ko batayein: StandardScaler kyun lagaya aur iska formula kya hai?",
        "StandardScaler scales features: z = (x - mean) / std. "
        "Without scaling, features with large ranges would overwhelm distance metrics in KNN and SVM.",
        "Sir, SVM aur KNN dono Euclidean distance par chalte hain. Agar hum scaling na karein toh jin features ki numerical value "
        "badi hogi woh doosre features ko suppress kar denge. StandardScaler sabhi 938 features ko zero mean aur unit variance "
        "par barabar karta hai."
    )

    # --- File 3: evaluate.py ---
    doc.add_heading("File 3: src/evaluate.py — Evaluation & Diagnostic Curves", level=2)
    p_f3 = doc.add_paragraph(
        "Yeh script held-out test set (347 images) par models evaluate karke research-grade outputs generate karti hai:\n"
        "• Confusion Matrix (outputs/confusion_matrix.png)\n"
        "• Classification Report (outputs/classification_report.txt)\n"
        "• Multi-class One-vs-Rest ROC Curves (outputs/roc_curves.png)\n"
        "• Learning Curves (outputs/learning_curves.png) — Overfitting/Underfitting test\n"
        "• Random Forest Feature Importance (outputs/feature_importance.png)\n"
        "• 3-Model Comparison Table (outputs/model_comparison_table.txt)"
    )
    p_f3.paragraph_format.line_spacing = 1.15

    # --- File 4: face_detector.py ---
    doc.add_heading("File 4: src/face_detector.py — Haar Cascade Face Detector", level=2)
    p_f4 = doc.add_paragraph(
        "OpenCV ka built-in classical Haar Cascade (`haarcascade_frontalface_default.xml`, Viola-Jones algorithm) "
        "use karta hai. Histogram equalization (`cv2.equalizeHist`) apply karta hai taake kam roshni mein bhi student "
        "ka chehra accurately detect ho sake, aur 20% margin ke sath face crop return karta hai."
    )

    # --- File 5: inference.py ---
    doc.add_heading("File 5: src/inference.py — Real-Time Prediction Engine", level=2)
    p_f5 = doc.add_paragraph(
        "`EngagementPredictor` class joblib se `models/best_model.joblib` load karti hai. Live video ke har face crop par "
        "wahi 938-dim feature extraction run karti hai aur `predict_proba()` se class probabilities return karti hai. "
        "Inference time sirf ~3 milliseconds hai!"
    )

    # --- File 6: app.py & index.html ---
    doc.add_heading("File 6: src/app.py & frontend/index.html — Live Dashboard & API", level=2)
    p_f6 = doc.add_paragraph(
        "Flask web server jo laptop webcam se frames read karke bounding boxes draw karta hai:\n"
        "• Green: Focused | Yellow: Confused | Red: Frustrated | Purple: Bored | Cyan: Drowsy | Gray: Looking Away\n"
        "Dashboard real-time class engagement score (%), session timer, aur end-of-session pedagogical recommendations "
        "(e.g., advising break if fatigue > 30%) dikhata hai."
    )

    # -------------------------------------------------------------
    # Section 4: Experimental Results & Embedded Plots
    # -------------------------------------------------------------
    doc.add_page_break()
    h4 = doc.add_heading("4. Experimental Results & Visual Diagnostic Plots", level=1)
    h4.paragraph_format.space_before = Pt(16)
    h4.paragraph_format.space_after = Pt(6)

    # Results Table
    res_table = doc.add_table(rows=4, cols=5)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_headers = ["Classifier", "Val Accuracy", "Test Accuracy", "Test F1-Score", "ROC AUC (Macro)"]
    for i, h in enumerate(r_headers):
        c = res_table.cell(0, i)
        c.text = h
        c.paragraphs[0].runs[0].font.bold = True
        set_cell_background(c, "1B365D")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    r_rows = [
        ("SVM (RBF Kernel, C=10) 🏆", "97.94%", "93.08%", "0.9306", "0.995"),
        ("Random Forest (n=200)", "98.23%", "91.35%", "0.9134", "0.992"),
        ("KNN (k=7, Distance-Weighted)", "98.23%", "90.78%", "0.9069", "0.989"),
    ]
    for r_idx, r_data in enumerate(r_rows):
        for c_idx, val in enumerate(r_data):
            cell = res_table.cell(r_idx + 1, c_idx)
            cell.text = val
            set_cell_background(cell, "EAEFF5" if r_idx == 0 else "FFFFFF")
            set_cell_margins(cell, 60, 60, 80, 80)

    doc.add_paragraph()

    # Embed Plots
    plots = [
        ("outputs/confusion_matrix.png", "Figure 1: Confusion Matrix on Held-Out Test Set (347 images)"),
        ("outputs/roc_curves.png", "Figure 2: Multi-Class One-vs-Rest ROC Curves (Mean AUC = 0.995)"),
        ("outputs/learning_curves.png", "Figure 3: Learning Curves (Sample Size vs Accuracy — No Overfitting)"),
        ("outputs/feature_importance.png", "Figure 4: Top 25 Most Informative Hand-Crafted Visual Features"),
    ]
    for img_rel, caption in plots:
        img_path = BASE_DIR / img_rel
        if img_path.exists():
            doc.add_paragraph().paragraph_format.space_before = Pt(8)
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(str(img_path), width=Inches(5.6))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.runs[0].font.size = Pt(9.5)
            p_cap.runs[0].font.italic = True
            p_cap.runs[0].font.color.rgb = RGBColor(80, 80, 80)
            doc.add_paragraph()

    # -------------------------------------------------------------
    # Section 5: The 10 Core ML Concepts
    # -------------------------------------------------------------
    doc.add_page_break()
    h5 = doc.add_heading("5. Mastery of 10 Core Machine Learning Concepts", level=1)
    h5.paragraph_format.space_before = Pt(16)
    h5.paragraph_format.space_after = Pt(6)

    ml_concepts = [
        ("1. Supervised Learning", "Labeled data (X, y) se mapping function seekhna jahan 2,277 facial crops ground-truth labels ke sath train huay."),
        ("2. Feature Engineering", "Raw pixels ko replace karke domain knowledge se HOG, LBP, HSV aur Haar-like features (938 dimensions) manually banana."),
        ("3. Feature Normalization (StandardScaler)", "Sabhi features ko mean=0 aur std=1 par normalize karna taake high-range features low-range features ko suppress na karein."),
        ("4. SVM (Support Vector Machine)", "Optimal hyperplane jo classes ke darmiyan margin maximize karta hai. RBF kernel non-linear data ko infinite dimensional space mein linearly separate karta hai."),
        ("5. Random Forest (Ensemble Learning)", "200 Decision Trees ka Bagging ensemble jo variance kam karta hai aur feature importance (MDI) batata hai."),
        ("6. K-Nearest Neighbors (KNN)", "Instance-based lazy learner jisme training phase nahi hota; test sample ke qareeb tareen k=7 samples distance-weighted vote karte hain."),
        ("7. Cross-Validation (5-Fold StratifiedKFold)", "Model validation technique jo har fold mein class distribution preserve karke overfitting aur train/test leakage rokti hai."),
        ("8. Overfitting vs Underfitting", "Learning curves se evaluate kiya gaya. Training aur validation score mein sirf 4.5% ka tight gap hai jo perfect generalization prove karta hai."),
        ("9. Evaluation Metrics", "Confusion Matrix, Precision (TP/(TP+FP)), Recall (TP/(TP+FN)), F1-Score (harmonic mean), aur One-vs-Rest ROC AUC."),
        ("10. Hyperparameter Tuning", "Systematic parameter comparison table (SVM C, RF n_estimators, KNN k) se champion hyperparameter select karna."),
    ]
    for c_title, c_desc in ml_concepts:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_before = Pt(4)
        p_c.paragraph_format.space_after = Pt(4)
        r1 = p_c.add_run(f"• {c_title}: ")
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(27, 54, 93)
        r2 = p_c.add_run(c_desc)

    # -------------------------------------------------------------
    # Section 6: Teacher Viva Defense Q&A
    # -------------------------------------------------------------
    doc.add_page_break()
    h6 = doc.add_heading("6. Teacher Viva / Defense Q&A (Top Tricky Questions)", level=1)
    h6.paragraph_format.space_before = Pt(16)
    h6.paragraph_format.space_after = Pt(6)

    viva_questions = [
        (
            "Sawal 1: Aapke project mein Machine Learning kahan hai? Kya yeh Deep Learning nahi hai?",
            "Isme koi deep learning ya neural network nahi hai. Humne CNN ki jagah manual Feature Engineering ki hai—HOG se edges, LBP se texture, HSV se color, aur Haar-like differences se face geometry extract karke 938 dimensions ka feature vector banaya hai. Phir scikit-learn ke classical models (SVM, Random Forest, KNN) train kiye hain."
        ),
        (
            "Sawal 2: Feature vector 938 dimensions ka breakdown batao.",
            "Har face crop 96x96 ka hota hai. HOG ke 800 features facial contours dete hain, LBP ke 90 features skin texture dete hain, HSV ke 32 features lighting-invariant color dete hain, aur 16 Haar features aankh, naak aur maathay ke contrast differences dete hain. Total = 800 + 90 + 32 + 16 = 938."
        ),
        (
            "Sawal 3: StandardScaler lagana kyun zaroori tha? Agar na lagate toh kya hota?",
            "SVM aur KNN dono Euclidean distance par depend karte hain. Agar StandardScaler na lagate toh jis feature ki numerical value badi hoti woh doosre features ko suppress kar deti. StandardScaler sabhi 938 features ko zero mean aur unit variance par scale karke equal weight deta hai."
        ),
        (
            "Sawal 4: SVM kyun sabse best perform kar raha hai?",
            "SVM maximum margin hyperplane dhoondta hai. Aur RBF (Radial Basis Function) kernel 938-dimensional space ko non-linear infinite Hilbert space mein map karta hai jahan engagement classes linearly separable ho jati hain. Isne test set par 93.08% accuracy aur 0.995 ROC AUC diya."
        ),
        (
            "Sawal 5: Overfitting aur Underfitting kaise check kiya?",
            "Humne evaluate.py mein Learning Curves plot kiye hain. Training curve aur Cross-Validation curve ke darmiyan sirf 4.5% ka gap hai. Agar training 100% hoti aur validation 60% hoti toh overfitting hoti; agar dono 60% hoti toh underfitting hoti. Hamara model well-regularized hai."
        ),
        (
            "Sawal 6: K-Fold ki jagah StratifiedKFold kyun use kiya?",
            "Standard K-Fold randomly split karta hai jis se kisi fold mein minority class kam ho sakti hai. StratifiedKFold guarantee karta hai ke har fold mein 6 classes ka percentage wahi rahe jo original dataset mein hai."
        ),
        (
            "Sawal 7: Face Detection ke liye kya use kiya hai?",
            "OpenCV ka classical Haar Cascade Classifier (haarcascade_frontalface_default.xml, Viola-Jones algorithm) jo integral images aur AdaBoost cascade par chalta hai. Yeh bhi purely classical ML approach hai."
        ),
        (
            "Sawal 8: System ko run kaise karte hain?",
            "1. Model train karne ke liye: python src/train_ml.py\n2. Graphs aur evaluation ke liye: python src/evaluate.py\n3. Live webcam dashboard ke liye: python src/app.py (aur browser mein http://localhost:5000 khulega)."
        ),
    ]

    for q, a in viva_questions:
        add_callout(doc, q, "Detailed technical answer available in section above.", a)

    # Save document
    doc.save(str(DOCX_PATH))
    print(f"\nSuccessfully generated Word Document at: {DOCX_PATH}")


if __name__ == "__main__":
    build_document()
