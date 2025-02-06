import base64
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
from importlib import import_module
import importlib.util

def mitigation(mitigations, dataframe, priv, target_name, download_url):
    report = {}
    pdf_buffers = {}

    for mitigate in mitigations:
        dataframe_copy = dataframe.copy()
        mitigation_name = mitigate["name"]
        priv_feature = priv["name"]
        priv_value = priv["value"]

        if mitigation_name == "FairGAN":
            print("FairGAN mitigation started...")
            from .FairGAN.main import mitigate
            # from FairGAN.main import mitigate
            statistics, base64_pdf = mitigate(dataframe_copy, target_name, priv_feature, priv_value, download_url)

        elif mitigation_name == "FairUS":
            print("FairUS mitigation started...")
            # from FairUS.main import mitigate
            from .FairUS.main import mitigate
            statistics, base64_pdf = mitigate(dataframe_copy, target_name, priv_feature, priv_value, download_url)

        elif mitigation_name == "FairSMOTE":
            print("FairSMOTE mitigation started...")
            # from FairSMOTE.main import mitigate
            from .FairSMOTE.main import mitigate
            statistics, base64_pdf = mitigate(dataframe_copy, target_name, priv_feature, priv_value, download_url)

        if statistics is not None and base64_pdf is not None:
            report[mitigation_name] = statistics
            pdf_buffers[mitigation_name] = base64.b64decode(base64_pdf)
        else:
            raise Exception("Mitigation failed")

    report_items = list(report.items())
    sorted_report_items = sorted(report_items, key=lambda x: x[1]["Upsampled"]["opportunity_difference"])

    # Merge PDFs according to the sorted order
    pdf_writer = PdfWriter()

    for item in sorted_report_items:
        mitigation_name = item[0]
        pdf_buffer = pdf_buffers[mitigation_name]
        pdf_reader = PdfReader(BytesIO(pdf_buffer))

        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

    # Write the combined PDF to a BytesIO object
    combined_pdf_buffer = BytesIO()
    pdf_writer.write(combined_pdf_buffer)
    combined_pdf_buffer.seek(0)

    # Encode the combined PDF to a base64 string
    combined_pdf_base64 = base64.b64encode(combined_pdf_buffer.read()).decode('utf-8')

    return  report, combined_pdf_base64

# import pandas as pd
# # testing
# dataframe = pd.read_csv(r'C:\Users\user\PycharmProjects\api\Backend\api\src\bias_eval\user_files_bias\dataset\bank.csv')
# priv = {"name": "housing", "value": "no"}
# target_name = "deposit"
# mitigations = [ {"name": "FairUS"}]
# report, pdf = mitigation(mitigations, dataframe, priv, target_name, 'gs://e2e-mabadata/Bias_data/save_unsampled/unsampled.csv')