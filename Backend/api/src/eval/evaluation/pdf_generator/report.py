import os
import json
from fpdf import FPDF
from datetime import datetime  # Importing datetime for timestamps
from .utils import plot_polygon, \
    plot_attack_performance_on_clean_model, \
    plot_defenses_average_ASR, plot_WB_table_weak_adversary, \
    plot_BB_table_weak_adversary, plot_Clean_task_performance_table  # Importing utility function for plotting rankings
# from Summary import full_summary  # Importing summary generation module
# import matplotlib.pyplot as plt
from math import pi
import numpy as np

import pandas as pd
from ....api.routers.helpers import get_from_db

PROJECT_ID = os.getenv("PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")

class ProfessionalPDF(FPDF):
    """
    Custom PDF class inheriting from FPDF to generate a professional-style report.
    """

    def __init__(self):
        ''' Constructor to initialize the PDF with specific settings '''
        super().__init__()
        print("create pdf object")
        self.set_auto_page_break(auto=True, margin=15)  # Set automatic page breaks with a margin of 15
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf')
        bold_font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed-Bold.ttf')
        print("Font path:", font_path)
        # Add fonts
        self.add_font('DejaVu', '', font_path, uni=True)
        self.add_font('DejaVu', 'B', bold_font_path, uni=True)
    def header(self):
        ''' Method to create the header of each page '''
        self.set_font('DejaVu', 'B', 12)  # Set font to bold DejaVu with size 12
        self.set_text_color(100, 100, 100)  # Set text color to a grey tone
        self.cell(0, 10, 'Model Evaluation Report', 0, 1, 'R')  # Create a right-aligned header with the report title
        self.line(10, 20, 200, 20)  # Draw a line below the header

    def footer(self):
        ''' Method to create the footer of each page '''
        self.set_y(-15)  # Position the footer 15 units from the bottom
        self.set_font('DejaVu', '', 8)  # Set font to standard DejaVu with size 8
        self.set_text_color(100, 100, 100)  # Set text color to a grey tone
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')  # Center-align the page number in the footer

    def chapter_title(self, title):
        ''' Method to create a chapter title with specific styling '''
        self.set_font('DejaVu', 'B', 18)  # Set font to bold DejaVu with size 18
        self.set_fill_color(155, 200, 255)  # Set background color to a light blue
        self.cell(0, 10, title, 0, 1, 'C', 1)  # Left-align the title with background fill
        self.ln(5)  # Add a line break after the title

    def chapter_sub_title(self, title):
        ''' Method to create a chapter subtitle with specific styling '''
        self.set_font('DejaVu', 'B', 16)  # Set font to bold DejaVu with size 16
        self.set_fill_color(200, 220, 255)  # Set background color to a light blue
        self.cell(0, 10, title, 0, 1, 'L', 1)  # Left-align the subtitle with background fill
        self.ln(5)  # Add a line break after the subtitle

    def bullet_point(self, text, indent=10):
        self.set_x(indent)  # Set indentation
        self.cell(10, 7, "-", 0, 0)
        self.multi_cell(0, 7, text)

    def bold_underline_title(self, title):
        self.set_font('Arial', 'BU', 12)  # 'B' for bold, 'U' for underline
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(4)
        self.set_font('Arial', '', 12)

    def bold_title(self, title):
        self.set_font('Arial', 'B', 12)  # 'B' for bold, 'U' for underline
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(4)
        self.set_font('Arial', '', 12)

    def chapter_body(self, body):
        ''' Method to add the main text content of a chapter '''
        self.set_font('DejaVu', '', 12)  # Set font to standard DejaVu with size 12
        self.multi_cell(0, 5, body)  # Add the body text with word wrapping
        self.ln()  # Add a line break after the body text

    def create_table(self, headers, data):
        ''' Method to create a table with headers and data '''
        self.set_font('DejaVu', 'B', 12)  # Set font to bold DejaVu with size 12 for headers
        self.set_fill_color(200, 220, 255)  # Set background color for headers to a light blue
        for header in headers:
            self.cell(40, 7, header, 1, 0, 'C', 1)  # Create a cell for each header, center-aligned with background fill
        self.ln()  # Add a line break after the header row
        self.set_font('DejaVu', '', 12)  # Set font to standard DejaVu with size 12 for data
        for i, row in enumerate(data):
            fill = False if i % 2 == 0 else True  # Alternate row fill color
            for item in row:
                self.cell(40, 7, str(item), 1, 0, 'C', fill)  # Create a cell for each data item, center-aligned
            self.ln()  # Add a line break after each row
        self.ln(5)  # Add an extra line break after the table
        


def create_cover_page(pdf):
    ''' Function to add a cover page to the PDF '''
    pdf.add_page()  # Add a new page for the cover
    pdf.set_font('DejaVu', 'B', 24)  # Set font to bold DejaVu with size 24 for the title
    pdf.cell(0, 40, 'AutoDefenceML', 0, 1, 'C')  # Center-align the title
    pdf.cell(0, 15, 'Model Evaluation Report', 0, 1, 'C')  # Center-align the title
    pdf.set_font('DejaVu', '', 14)  # Set font to standard DejaVu with size 14 for the date
    pdf.cell(0, 20, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}', 0, 1,
             'C')  # Center-align the generation date

    pdf.chapter_title('About Us')
    pdf.chapter_body(
        "This report is developed by Ben-Gurion University in collaboration with the Israel National Cyber Directorate. "
        "The project focuses on analyzing and identifying vulnerabilities in machine learning models against "
        "adversarial attacks. Our tool offers three key features:\n"
        "   1) Assess the security of trained models, providing suggestions for defenses.\n"
        "   2) Evaluate training datasets to detect security risks and recommend data cleaning techniques.\n"
        "   3) Examine bias within both the model and the dataset.\n\n"
        "The tool simulates known adversarial attacks to proactively identify weaknesses in models, and tests defenses to "
        "determine the most robust protection strategies. Based on the results, it suggests improvements and may even generate "
        "a more resilient version of the model. Additionally, the tool applies model-independent techniques to analyze datasets "
        "for anomalies, biases, and other issues, delivering a detailed report along with a cleaned version of the data."
    )
    # Move the cursor down to the bottom of the page
    pdf.image(os.path.join(os.path.dirname(__file__),'ben_gurion_logo.png'), x=25, y=200, w=70)  # Ben Gurion University logo
    pdf.image(os.path.join(os.path.dirname(__file__),'incd_logo.png'), x=105, y=200, w=80)


def generate_pdf(response_data, job_id):

    report_request = get_from_db(PROJECT_ID, FIRESTORE_DB, "Requests", job_id)
    # File paths for the response and attack map JSON files
    # response_file_path = '667e5e4ec8514ca23f2eef5c-21b5c963-6a8e-4629-98cb-437639d5aaba.json'
    # attack_map_file_path = 'attack_defense_map.json'
    # response_file_path = 'Results_JSON_yisroel.json'
    # Initialize the PDF report
    pdf = ProfessionalPDF()
    create_cover_page(pdf)
    pdf.add_page()

    ############################################## Introduction To Terms ##################################
    pdf.chapter_title('Introduction to Key Terms')

    # pdf.add_bold_paragraph()
    pdf.chapter_body("In this report, we will explore a variety of concepts and metrics used to evaluate model performance, both in terms of its behavior on clean data and its robustness against adversarial attacks.\nBelow is an overview of the key terms and metrics discussed throughout the document:")
    # Clean Task Performance (Bold)
    pdf.bold_underline_title('Clean Task Performance:')
    pdf.chapter_body("Refers to the performance of the model on the clean, unaltered data provided by the user. This is typically measured by standard performance metrics such as accuracy, AUC, precision, and recall.")
    # ASR (Bold)
    pdf.bold_underline_title("ASR (Attack Success Rate):")
    pdf.chapter_body("The rate at which adversarial attacks succeed in compromising the model's predictions. A higher ASR indicates that the model is more vulnerable to the specific attack.")
    # AUC (Bold)
    pdf.bold_underline_title("AUC (Area Under the Curve):")
    pdf.chapter_body("A metric used to evaluate the accuracy of the model. It represents the likelihood that the model will rank a randomly chosen positive instance higher than a randomly chosen negative one.")
    # BALANCED Choice (Bold)
    pdf.bold_underline_title("BALANCED Choice 5%:")
    pdf.chapter_body("Refers to a defense mechanism that effectively mitigates adversarial attacks while maintaining a strong performance on the clean task. A 'balanced' defense ensures that the clean task performance is not degraded by more than 5% while providing robust protection against attacks.")
    # Model Accuracy (ACC)
    pdf.bold_underline_title("Model Accuracy (ACC):")
    pdf.chapter_body("Accuracy is one of the fundamental performance metrics for evaluating machine learning models. It measures the proportion of correctly classified instances out of the total number of instances. A higher accuracy reflects better model performance on the given task.")


    # Introduction section of the report
    pdf.add_page()

    # ##TODO:: find a json file that has this information regarding the dataset and model.
    # with open(response_file_path, 'r') as file:
    #     response_data = json.load(file)  # Load the response data from the file
    #
    with open(os.path.join(os.path.dirname(__file__),'updated_attack_defense_map.json'), 'r') as file:
        attack_mapping = json.load(file)

    # # Extract dataset information
    # dataset_info = response_data.get('dataset_info', {})  # Get dataset information, defaulting to an empty dictionary
    # dataset_name = dataset_info.get('name', 'Unknown')  # Get the dataset name, defaulting to 'Unknown'
    # dataset_type = dataset_info.get('type', 'Unknown')

    ############################################## Summary ##################################

    pdf.chapter_title('Executive Summary')
    pdf.chapter_body(
        """In this chapter, we provide a comprehensive overview of the evaluated model, focusing on its general performance and behavior. This includes a detailed analysis of its clean task performance, which refers to how well the model performs on the clean, unaltered dataset supplied by the user. By presenting key metrics and general statistics about the model, we aim to give a clear picture of its behavior in standard conditions.

    The clean task performance is especially important, as it serves as a baseline for understanding how the model behaves in the absence of adversarial attacks or noise. It is critical that users review this section carefully to ensure that the model performs as expected on the data they provided. This is an opportunity for the user to validate whether the model’s predictions and outcomes align with their domain-specific expectations. Any discrepancies between the expected and actual performance should be noted, as they may impact the subsequent evaluation under adversarial conditions. This review can help ensure that the model is functioning correctly before proceeding with more complex evaluations.
        """)

    pdf.chapter_sub_title('Run Configurations')
    left_column_x = 10  # X position for the left column
    right_column_x = 110  # X position for the right column (adjust as needed to fit the page)

    # Set the y position to align the sections properly
    start_y = pdf.get_y()

    try:
        # Dataset statistics
        number_of_samples = response_data.get("dataset_statistics", {}).get("number_of_samples", None)
        # Model dimensions
        dim_data = report_request.get("request", {}).get("ml_model", {}).get("dim", {})
        number_of_features = dim_data.get("input", None)
        number_of_classes = dim_data.get("num_classes", None)
        number_of_clip_values = dim_data.get("clip_values", None)
        # Model metadata
        meta_data = report_request.get("request", {}).get("ml_model", {}).get("meta", {})
        framework_name = meta_data.get("framework", None)  # ישירות מ-meta_data
        ml_task = meta_data.get("ml_type", None)  # ישירות מ-meta_data
        # Model file path handling
        parameters = meta_data.get("parameters", {})
        model_path = parameters.get("path", None)
        model_file_name = model_path.split('/')[-1] if model_path else None

    except Exception as e:
        print(f"Error parsing data: {e}")

    # Left Column - Dataset Section
    pdf.set_xy(left_column_x, start_y)
    pdf.bold_underline_title('Dataset: ')
    # pdf.bullet_point("Tabular: boolean")
    pdf.bullet_point(f"Number of Samples: {number_of_samples}")
    pdf.bullet_point(f"Number of Features: {number_of_features}")
    pdf.bullet_point(f"Number of Classes: {number_of_classes}")
    pdf.bullet_point(f"Clip Values: {number_of_clip_values}")

    # Right Column - Model Section
    pdf.set_xy(right_column_x, start_y)  # Move to the right side for the Model section
    pdf.bold_underline_title('Model:')
    pdf.bullet_point(f"Framework: {framework_name}", indent=right_column_x)
    pdf.bullet_point(f"File: {model_file_name}", indent=right_column_x)
    pdf.bullet_point(f"ML Task: {ml_task}", indent=right_column_x)

    # Continue with additional content if needed
    pdf.chapter_body('\n')

    ## TODO:: enter here the details of the performance from the report. only The clean.

    ############################################## Clean Data Perofrmance ##################################

    pdf.chapter_sub_title("The Model's Performance on Clean Data")
    pdf.chapter_body(
        'Here, we are presenting your model performance, on the data ytou have suplied. Please, Check that everything is on place.')

    ctp = plot_Clean_task_performance_table(response_data)
    current_y = pdf.get_y()
    pdf.image(ctp, x=10, y=current_y, w=130)

    pdf.add_page()
    ############################################## Security Card ##################################

    pdf.chapter_title('Security Report Card')
    pdf.chapter_body(
        """In this chapter, we present the results of the adversarial attacks conducted on the clean model you have supplied. These results provide a detailed look at how vulnerable your model is to different types of attacks, offering insights into its robustness.

     The performance of each attack is measured using the Attack Success Rate (ASR), which indicates the percentage of successful adversarial manipulations that fooled the model. We have organized the results to clearly distinguish between White-Box (WB) and Black-Box (BB) attacks, allowing you to assess and compare the model's resistance under different attack scenarios.

     This distinction is critical for understanding how your model performs under more or less restrictive attack conditions. The report is structured in a way that enables you to easily identify which attacks pose the greatest risk, helping you tailor any future defenses to your model's specific vulnerabilities and needs. Reviewing these results will give you the insights necessary to make informed decisions about strengthening your model’s robustness and ensuring it aligns with your operational requirements."
        """)

    wb_attacks_sorted, bb_attacks_sorted, plot_ASR_path = plot_attack_performance_on_clean_model(response_data['model_without_defence']['with_attack'], attack_mapping, pdf)
    polygons_path = plot_polygon(wb_attacks_sorted, bb_attacks_sorted, response_data, pdf)

    image_width = 0
    if wb_attacks_sorted:
        image_width += 95
    if bb_attacks_sorted:
        image_width += 95

    current_y = pdf.get_y()
    image1_height = 90  # Estimate height of the first image in PDF units
    pdf.image(plot_ASR_path, x=10, y=current_y + 5, w=image_width)
    current_y = pdf.get_y() + image1_height

    pdf.add_page()  # Estimate height of the second image in PDF units

    pdf.chapter_sub_title("Overview")
    pdf.chapter_body(
        "Overview in terms of:\n   1.Clean ACC: The model's clean task performance.\n   2.Clean AUC: The Model's clean AUC score.\n   3.Mean AFR: The average AFR.\n   4.Min AFR: The min AFR.")
    pdf.ln(10)
    current_y = pdf.get_y()
    pdf.image(polygons_path, x=10, y=current_y + 10, w=image_width)
    pdf.ln(120)
    # pdf.image('Clean_model_attack_performance/WB_BB_polygons.png', x=10 + 50 + 10, y=50, w=90)

    # pdf.add_page()
    pdf.chapter_sub_title("Quick Summary")
    if len(wb_attacks_sorted) > 0:
        pdf.bold_title('White Box Attacks:')
        pdf.bullet_point(f"The average ASR on your model was {np.mean([j for i, j in wb_attacks_sorted]):.4f}")
        pdf.bullet_point(f"The most Powerful attack was {wb_attacks_sorted[0][0]} with an ASR of {wb_attacks_sorted[0][1]:.4f}")
    if len(bb_attacks_sorted) > 0:
        pdf.bold_title('')
        pdf.bold_title('Black Box Attacks:')
        pdf.bullet_point(f"The average ASR on your model was {np.mean([j for i, j in bb_attacks_sorted]):.4f}")
        pdf.bullet_point(f"The most Powerful attack was {bb_attacks_sorted[0][0]} with an ASR of {bb_attacks_sorted[0][1]:.4f}")

    ############################################## Defense Recomendation - Weak ##################################

    pdf.add_page()
    pdf.chapter_title('Defense Recommendation')
    pdf.chapter_body(
        "In this section, we present the results of the defenses that were tested on your model, showcasing how each one performed in mitigating adversarial attacks. Our goal is to provide insights that will help you choose the most effective defenses to protect your model from potential threats. By analyzing these results, you will be able to make informed decisions about which defense strategies best suit your model's needs.")
    pdf.chapter_sub_title("Weak Adversary (Static)")

    pdf.chapter_body("""In this chapter, we present an evaluation of the different defense mechanisms applied to your model, showcasing their ability to counter the adversarial examples generated by various attacks on your clean model. The aim of this section is to demonstrate how well each defense strategy performed in mitigating the impact of these adversarial attacks.

    The plot below represents the Average Attack Success Rate (ASR) for each defense, summarizing their effectiveness against all the attacks within the same scenario—whether White-Box (WB) or Black-Box (BB). This visualization allows you to easily compare the defenses' performance and identify which strategies were more effective in reducing the model’s vulnerability in each scenario.

    By examining these results, you can gain insight into which defenses offer the most robust protection for your model, and how well they perform in specific attack environments. This information will help guide your decision-making when selecting or fine-tuning defense mechanisms to optimize your model's resilience to adversarial threats""")

    average_defenses_plot_path = plot_defenses_average_ASR(response_data, attack_mapping)
    current_y = pdf.get_y()
    pdf.image(average_defenses_plot_path, x=10, y=current_y + 5, w=image_width)

    WB_Table = plot_WB_table_weak_adversary(response_data, attack_mapping)
    pdf.ln(95)  # Add some spacing before the next section if needed
    pdf.add_page()
    pdf.chapter_sub_title(f'Detailed Defance Report')
    # pdf.chapter_body("All key information is summarized in the following tables.")
    current_y = pdf.get_y()
    pdf.image(WB_Table, x=10, y=current_y, w=190)
    pdf.ln(95)
    BB_Table = plot_BB_table_weak_adversary(response_data, attack_mapping)
    # current_y = pdf.get_y()
    # pdf.image(BB_Table, x=10, y=current_y, w=190)

    image_height = 95  # You can adjust this depending on the image size
    page_height = pdf.h - pdf.b_margin  # Get the usable height of the page

    # Add BB Table
    if BB_Table:
        current_y = pdf.get_y()
        if current_y + image_height > page_height:  # Check if there's enough space for the BB table
            pdf.add_page()  # Add a new page if needed
            pdf.image(BB_Table, x=10, y=10, w=image_width)
        else:
            pdf.image(BB_Table, x=10, y=current_y, w=image_width)

    if response_data.get('optimize_attacks_on_defense_reports', False):
        ############################################## Defense Recomendation - Strong ##################################

        pdf.add_page()

        pdf.chapter_sub_title("Strong Adversary (Optimized)")
        pdf.chapter_body(
            f"In this chapter, we present an evaluation of the different defense mechanisms applied to your model, showcasing their ability to counter the adversarial examples generated by various attacks when defenses are in place. This section aims to evaluate the robustness of defenses against attackers who adapt their strategies based on the defense in place.")
        average_defenses_plot_path_Strong = plot_defenses_average_ASR(response_data, attack_mapping, strong_adv=True)
        current_y = pdf.get_y()
        pdf.image(average_defenses_plot_path_Strong, x=10, y=current_y + 5, w=190)

        pdf.add_page()

        pdf.chapter_sub_title(f'Quick Summary')
        pdf.chapter_body("All key information is summarized in the following tables.")

        WB_Table_optimized = plot_WB_table_weak_adversary(response_data, attack_mapping, strong_adv=True)
        # pdf.ln(95)  # Add some spacing before the next section if needed
        current_y = pdf.get_y()
        pdf.image(WB_Table_optimized, x=10, y=current_y + 20, w=190)
        pdf.ln(95)

        BB_Table_optimized = plot_BB_table_weak_adversary(response_data, attack_mapping, strong_adv=True)
        # current_y = pdf.get_y()
        # pdf.image(BB_Table_optimized, x=10, y=current_y, w=190)

        image_height = 95  # You can adjust this depending on the image size
        page_height = pdf.h - pdf.b_margin  # Get the usable height of the page

        # Add BB Table
        current_y = pdf.get_y()
        if current_y + image_height > page_height:  # Check if there's enough space for the BB table
            pdf.add_page()  # Add a new page if needed
            pdf.image(BB_Table_optimized, x=10, y=10, w=190)
        else:
            pdf.image(BB_Table_optimized, x=10, y=current_y, w=190)
        pdf.ln(95)

    # absulute path to the pdf
    path = os.path.join(os.path.dirname(__file__), 'model_performance_report.pdf')
    pdf.output(path)
    # reurn bytes string of pdf
    ##TODO:: return the pdf as bytes string
    # Encode the combined PDF to a base64 string
    import base64
    with open(path, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read()).decode('utf-8')
    # from io import BytesIO
    # pdf_bytes = BytesIO()
    # pdf_data = pdf_bytes.getvalue()
    #
    # # pdf.output(combined_pdf_buffer)
    return encoded_string
    # return base64.b64encode(combined_pdf_buffer.getvalue()).decode('utf-8')

# test
# path = os.path.join(os.path.dirname(__file__), '667e5e4ec8514ca23f2eef5c-070a5e6c-796c-440c-a022-fafd618ef9ec.json')
# with open(path, 'r') as file:
#     response_data = json.load(file)  # Load the response data from the file
# bytes = generate_pdf(response_data)
# print(bytes)
# import base64
# decoded_pdf = base64.b64decode(bytes)
# #open this pdf
# with open('output.pdf', 'wb') as f:
#     f.write(decoded_pdf)
# # import base64
# # #bas264 encoded string:
# # encoded_str = base64.b64encode(bytes).decode('utf-8')
# # # str = base64.b64encode(bytes).decode('utf-8')
# # # print(str)
# # decoded_bytes = base64.b64decode(encoded_str)
# #
# # # Optionally save the decoded bytes as a file to verify it works as expected
# # with open('output.pdf', 'wb') as f:
# #     f.write(decoded_bytes)
