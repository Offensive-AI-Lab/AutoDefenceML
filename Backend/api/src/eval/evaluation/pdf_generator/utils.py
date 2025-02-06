import matplotlib.pyplot as plt
from math import pi
import numpy as np
import pandas as pd
import os

def plot_attack_performance_on_clean_model(response_data, attack_mapping, pdf):
    attack_rates = {}
    for attack_name, attack_data in response_data.items():
        if not isinstance(attack_data, dict):
            continue
        attack_rates[attack_name] = attack_data.get('attack_success_rate', None)

    wb_attacks = []
    bb_attacks = []

    for attack_name, ASR in attack_rates.items():
        if ASR is not None and attack_name in attack_mapping.get('attacks', {}):
            if attack_mapping['attacks'][attack_name]['assumption'] == 'WB':  # White-box attack
                wb_attacks.append((attack_name, attack_rates[attack_name]))
            elif attack_mapping['attacks'][attack_name]['assumption'] == "BB":  # Black-box attack
                bb_attacks.append((attack_name, attack_rates[attack_name]))
    
    # Sort both WB and BB attacks by success rate (descending order)
    wb_attacks_sorted = sorted(wb_attacks, key=lambda x: x[1], reverse=True)
    bb_attacks_sorted = sorted(bb_attacks, key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    wb_attack_names = [attack[0] for attack in wb_attacks_sorted]
    wb_success_rates = [attack[1] for attack in wb_attacks_sorted]

    bb_attack_names = [attack[0] for attack in bb_attacks_sorted]
    bb_success_rates = [attack[1] for attack in bb_attacks_sorted]

    # אם יש רק WB
    if wb_attacks and not bb_attacks:
        fig, ax1 = plt.subplots(figsize=(7, 6))
        ax1.bar(wb_attack_names, wb_success_rates, color='skyblue')
        ax1.set_ylabel('ASR', fontsize=16)
        ax1.set_title('White-box', fontsize=20)
        ax1.tick_params(axis='x', rotation=45, labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
    # אם יש רק BB
    elif bb_attacks and not wb_attacks:
        fig, ax1 = plt.subplots(figsize=(7, 6))
        ax1.bar(bb_attack_names, bb_success_rates, color='lightcoral')
        ax1.set_ylabel('ASR', fontsize=16)
        ax1.set_title('Black-box', fontsize=20)
        ax1.tick_params(axis='x', rotation=45, labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
    # אם יש את שניהם
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # Plot White-box Attacks
        if wb_attacks:
            ax1.bar(wb_attack_names, wb_success_rates, color='skyblue')
            ax1.set_ylabel('ASR', fontsize=16)
            ax1.set_title('White-box', fontsize=20)
            ax1.tick_params(axis='x', rotation=45, labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
        # Plot Black-box Attacks
        if bb_attacks:
            ax2.bar(bb_attack_names, bb_success_rates, color='lightcoral')
            ax2.set_ylabel('ASR', fontsize=16)
            ax2.set_title('Black-box', fontsize=20)
            ax2.tick_params(axis='x', rotation=45, labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the combined plot as a single image
    combined_plot_dir = os.path.join(os.path.dirname(__file__), 'Clean_model_attack_performance')
    combined_plot_path = os.path.join(combined_plot_dir, 'combined_attack_success_rates.png')

    # Ensure the directory exists
    os.makedirs(combined_plot_dir, exist_ok=True)

    # Save the plot
    plt.savefig(combined_plot_path)
    plt.close()

    print("Attacks on Clean Model Performance is generated SUCCESSFULLY.")
    return wb_attacks_sorted, bb_attacks_sorted, combined_plot_path

def calculate_polygon_area(values, angles):
    """
    Calculate the area of a polygon using the shoelace formula in polar coordinates.
    
    Args:
        values: List of radii values
        angles: List of angles in radians
    
    Returns:
        area: Area of the polygon
    """
    x = [r * np.cos(theta) for r, theta in zip(values[:-1], angles[:-1])]
    y = [r * np.sin(theta) for r, theta in zip(values[:-1], angles[:-1])]
    
    area = 0.0
    j = len(x) - 1
    
    for i in range(len(x)):
        area += (x[j] + x[i]) * (y[j] - y[i])
        j = i
    
    return abs(area) / 2.0

def plot_polygon(wb_attacks_sorted, bb_attacks_sorted, response_data, pdf):
    labels = ['3.Mean-AFR', '4.Min-AFR', '1.Clean ACC', '2.Clean AUC']
    mean_afr_wb , min_afr_wb , mean_afr_bb , min_afr_bb, WB_values, BB_values= None, None, None, None , None, None
    if len(wb_attacks_sorted) > 0:
        mean_afr_wb = 1 - np.mean([v for k, v in wb_attacks_sorted])
        min_afr_wb = 1 - np.min([v for k, v in wb_attacks_sorted])
    clean_acc = response_data['model_without_defence']['without_attack']['accuracy']
    clean_auc = response_data['model_without_defence']['without_attack']['auc']

    if len(bb_attacks_sorted) > 0:
        mean_afr_bb = np.mean([v for k, v in bb_attacks_sorted])
        min_afr_bb = np.min([v for k, v in bb_attacks_sorted])

    if mean_afr_wb:
        WB_values = [mean_afr_wb, min_afr_wb, clean_acc, clean_auc]
    if mean_afr_bb:
        BB_values = [mean_afr_bb, min_afr_bb, clean_acc, clean_auc]

    num_vars = len(labels)

    # Compute angle for each axis in the plot (equal spacing in a radar chart)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the radar chart

    # אם יש רק WB
    if WB_values and not BB_values:
        fig, ax1 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        WB_values += WB_values[:1]
        ax1.plot(angles, WB_values, linewidth=2, linestyle='solid', label='White-box Attacks', color='b')
        ax1.fill(angles, WB_values, 'b', alpha=0.3)
        ax1.set_title('White-box Attacks', size=15, color='black')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, size=12)
        # Add area text in the center
        wb_area = calculate_polygon_area(WB_values, angles) / 2
        ax1.text(0, 0, f'Health Score: {wb_area:.2f}', 
                ha='center', va='center',
                fontsize=16,
                bbox=dict(facecolor='white', alpha=0.7))

    # אם יש רק BB
    elif BB_values and not WB_values:
        fig, ax1 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        BB_values += BB_values[:1]
        ax1.plot(angles, BB_values, linewidth=2, linestyle='solid', label='Black-box Attacks', color='r')
        ax1.fill(angles, BB_values, 'r', alpha=0.3)
        ax1.set_title('Black-box Attacks', size=15, color='black')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, size=12)
        # Add area text in the center
        bb_area = calculate_polygon_area(BB_values, angles) / 2
        ax1.text(0, 0, f'Health Score: {bb_area:.2f}', 
                ha='center', va='center',
                fontsize=16,
                bbox=dict(facecolor='white', alpha=0.7))

    # אם יש את שניהם
    else:  # במקרה זה בהכרח יש את שניהם
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

        # Plot WB
        WB_values += WB_values[:1]
        ax1.plot(angles, WB_values, linewidth=2, linestyle='solid', label='White-box Attacks', color='b')
        ax1.fill(angles, WB_values, 'b', alpha=0.3)
        ax1.set_title('White-box Attacks', size=15, color='black')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, size=12)
        wb_area = calculate_polygon_area(WB_values, angles) / 2
        ax1.text(0, 0, f'Health Score: {wb_area:.2f}', 
                ha='center', va='center',
                fontsize=16,
                bbox=dict(facecolor='white', alpha=0.7))

        # Plot BB
        BB_values += BB_values[:1]
        ax2.plot(angles, BB_values, linewidth=2, linestyle='solid', label='Black-box Attacks', color='r')
        ax2.fill(angles, BB_values, 'r', alpha=0.3)
        ax2.set_title('Black-box Attacks', size=15, color='black')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, size=12)
        bb_area = calculate_polygon_area(BB_values, angles) / 2
        ax2.text(0, 0, f'Health Score: {bb_area:.4f}', 
                ha='center', va='center',
                fontsize=16,
                bbox=dict(facecolor='white', alpha=0.7))

    # Adjust the layout and save the figure
    plt.tight_layout()  # Adjust spacing to prevent overlap
    combined_plot_dir = os.path.join(os.path.dirname(__file__), 'Clean_model_attack_performance')
    combined_plot_path = os.path.join(combined_plot_dir, 'WB_BB_polygons.png')

    # Ensure the directory exists
    os.makedirs(combined_plot_dir, exist_ok=True)

    # Save the plot
    plt.savefig(combined_plot_path)
    plt.close()

    return combined_plot_path


# def plot_defenses_average_ASR(response_data, attack_mapping, strong_adv=False):
#     defenses_avg_ASR_wb = {}
#     defenses_avg_ASR_bb = {}
#
#     if strong_adv:
#         section = 'optimize_attacks_on_defense_reports'
#         folder = 'Strong Adversary'
#     else:
#         section = 'defense_evaluation'
#         folder = 'Weak Adversary'
#
#     # Loop through the JSON to compute average ASR for WB and BB attacks
#     for defense_name, defense_data in response_data.items():
#         if defense_name == 'clean_model_evaluation':
#             continue
#
#         wb_attack_rates = []
#         bb_attack_rates = []
#
#         for attack_name, attack_data in defense_data.get(section, {}).items():
#             attack_success_rate = attack_data['attack_success_rate']
#             if attack_success_rate is not None:
#                 if attack_mapping['attacks'][attack_name]['assumption'] == 'WB':  # White-box attack
#                     wb_attack_rates.append(attack_success_rate)
#                 elif attack_mapping['attacks'][attack_name]['assumption'] == "BB":  # Black-box attack
#                     bb_attack_rates.append(attack_success_rate)
#
#         # Calculate the average for WB and BB if there are any attacks
#         if wb_attack_rates:
#             defenses_avg_ASR_wb[defense_name] = np.mean(wb_attack_rates)
#         if bb_attack_rates:
#             defenses_avg_ASR_bb[defense_name] = np.mean(bb_attack_rates)
#
#     # Get defense names that have both WB and BB attacks
#     defenses_wb = list(defenses_avg_ASR_wb.keys())
#     defenses_bb = list(defenses_avg_ASR_bb.keys())
#     avg_wb_rates = [defenses_avg_ASR_wb[defense] for defense in defenses_wb]
#     avg_bb_rates = [defenses_avg_ASR_bb[defense] for defense in defenses_bb]
#
#     # Create subplots: One for WB, one for BB
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#
#     # Plot WB Attack Success Rates
#     ax1.bar(defenses_wb, avg_wb_rates, color='skyblue')
#     ax1.set_xlabel('Defenses')
#     ax1.set_ylabel('Average ASR')
#     ax1.set_title('Average WB Attack Success Rate by Defense', size=15)
#     ax1.set_xticklabels(defenses_wb, rotation=45, ha='right', size=12)
#
#     # Plot BB Attack Success Rates
#     ax2.bar(defenses_bb, avg_bb_rates, color='lightcoral')
#     ax2.set_xlabel('Defenses')
#     ax2.set_title('Average BB Attack Success Rate by Defense', size=15)
#     ax2.set_xticklabels(defenses_bb, rotation=45, ha='right', size=12)
#
#     # Adjust layout to avoid overlap
#     plt.tight_layout()
#
#     path = f'{folder}/WB_BB_average.png'
#     plt.savefig(path)
#     plt.close()
#     return path

def plot_defenses_average_ASR(response_data, attack_mapping, strong_adv=False):
    defenses_avg_ASR_wb = {}
    defenses_avg_ASR_bb = {}

    if strong_adv:
        section = 'optimize_attacks_on_defense_reports'
        folder = 'Strong Adversary'
    else:
        section = 'defense_evaluation'
        folder = 'Weak Adversary'

    # Loop through the JSON to compute average ASR for WB and BB attacks
    for defense_name, defense_data in response_data.get('model_with_defence', {}).items():

        wb_attack_rates = []
        bb_attack_rates = []

        for attack_name, attack_data in defense_data.get(section, {}).items():
            attack_success_rate = attack_data['attack_success_rate']
            if attack_success_rate is not None:
                if attack_mapping['attacks'][attack_name]['assumption'] == 'WB':  # White-box attack
                    wb_attack_rates.append(attack_success_rate)
                elif attack_mapping['attacks'][attack_name]['assumption'] == "BB":  # Black-box attack
                    bb_attack_rates.append(attack_success_rate)

        # Calculate the average for WB and BB if there are any attacks
        if wb_attack_rates:
            defenses_avg_ASR_wb[defense_name] = np.mean(wb_attack_rates)
        if bb_attack_rates:
            defenses_avg_ASR_bb[defense_name] = np.mean(bb_attack_rates)

    # Get defense names that have both WB and BB attacks
    defenses_wb = list(defenses_avg_ASR_wb.keys())
    defenses_bb = list(defenses_avg_ASR_bb.keys())
    avg_wb_rates = [defenses_avg_ASR_wb[defense] for defense in defenses_wb]
    avg_bb_rates = [defenses_avg_ASR_bb[defense] for defense in defenses_bb]

    # אם יש רק WB
    if defenses_wb and not defenses_bb:
        fig, ax1 = plt.subplots(figsize=(7, 6))
        x_positions = range(len(defenses_wb))
        ax1.bar(x_positions, avg_wb_rates, color='skyblue')
        ax1.set_xlabel('Defenses')
        ax1.set_ylabel('Average ASR')
        ax1.set_title('White Box: Average Attack Success Rate by Defense', size=15)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(defenses_wb, rotation=45, ha='right', size=12)
    # אם יש רק BB
    elif defenses_bb and not defenses_wb:
        fig, ax1 = plt.subplots(figsize=(7, 6))
        x_positions = range(len(defenses_bb))
        ax1.bar(x_positions, avg_bb_rates, color='lightcoral')
        ax1.set_xlabel('Defenses')
        ax1.set_ylabel('Average ASR')
        ax1.set_title('Black Box: Average BB Attack Success Rate by Defense', size=15)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(defenses_bb, rotation=45, ha='right', size=12)
    # אם יש את שניהם
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        # Plot WB Attack Success Rates
        x_positions_wb = range(len(defenses_wb))
        ax1.bar(x_positions_wb, avg_wb_rates, color='skyblue')
        ax1.set_xlabel('Defenses')
        ax1.set_ylabel('Average ASR')
        ax1.set_title('White Box: Average Attack Success Rate by Defense', size=15)
        ax1.set_xticks(x_positions_wb)
        ax1.set_xticklabels(defenses_wb, rotation=45, ha='right', size=12)
        # Plot BB Attack Success Rates
        x_positions_bb = range(len(defenses_bb))
        ax2.bar(x_positions_bb, avg_bb_rates, color='lightcoral')
        ax2.set_xlabel('Defenses')
        ax2.set_title('Black Box: Average BB Attack Success Rate by Defense', size=15)
        ax2.set_xticks(x_positions_bb)
        ax2.set_xticklabels(defenses_bb, rotation=45, ha='right', size=12)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    dir = os.path.join(os.path.dirname(__file__), folder)
    path = os.path.join(dir, 'WB_BB_average.png')
    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    plt.savefig(path)
    plt.close()
    return path

def plot_WB_table_weak_adversary(response_data, attack_mapping, strong_adv=False):
    defenses_avg_ASR_wb = {}
    defenses_clean_acc = {}

    if strong_adv:
        section = 'optimize_attacks_on_defense_reports'
        folder = 'Strong Adversary'
    else:
        section = 'with_attack'  # שינוי ראשון - שינוי החלק הרלוונטי לפי המבנה החדש
        folder = 'Weak Adversary'

    dir = os.path.join(os.path.dirname(__file__), folder)
    path = os.path.join(dir, 'WB_Table.png')
    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Get the clean model evaluation performance (baseline performance)
    baseline_clean_accuracy = response_data['model_without_defence']['without_attack'].get('accuracy', None)
    
    # Loop through the JSON to compute average ASR for WB attacks and Clean Accuracy
    for defense_name, defense_data in response_data.get('model_with_defence', {}).items():
        wb_attack_rates = []

        # Collect ASR for WB attacks
        for attack_name, attack_data in defense_data.get(section, {}).items():
            attack_success_rate = attack_data.get('attack_success_rate')  # שינוי שני - הוספת get
            if attack_success_rate is not None and attack_name in attack_mapping.get('attacks', {}) and attack_mapping['attacks'][attack_name]['assumption'] == 'WB':
                wb_attack_rates.append(attack_success_rate)

        # Calculate the average for WB if there are any WB attacks
        if wb_attack_rates:
            defenses_avg_ASR_wb[defense_name] = sum(wb_attack_rates) / len(wb_attack_rates)

        # Collect clean accuracy for each defense
        clean_acc = defense_data['without_attack'].get('accuracy', None)
        if clean_acc is not None:
            defenses_clean_acc[defense_name] = clean_acc

    if len(defenses_avg_ASR_wb) == 0:  # The dictionary is empty
        # Create a plot with the message "No White Box Attacks"
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)  # Set x-limits for centering text
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, 'No White Box Attacks', fontsize=24, ha='center', va='center', color='black')
        ax.set_axis_off()  # Hide the axes
        plt.savefig(path)  # Save the plot as an image
        plt.close()
        return path

    # Find the defense with the lowest ASR for "Security-First" (WB attacks)
    min_asr_defense = min(defenses_avg_ASR_wb, key=defenses_avg_ASR_wb.get)
    min_asr_value = defenses_avg_ASR_wb[min_asr_defense]
    min_asr_clean_acc = defenses_clean_acc.get(min_asr_defense, 'N/A')
    min_asr_params = response_data['model_with_defence'][min_asr_defense].get('params', 'N/A')  # שינוי שלישי - תיקון הנתיב

    # Find the defense with the highest clean accuracy for "Performance-First"
    max_clean_acc_defense = max(defenses_clean_acc, key=defenses_clean_acc.get)
    max_clean_acc_value = defenses_clean_acc[max_clean_acc_defense]
    max_clean_acc_asr = defenses_avg_ASR_wb.get(max_clean_acc_defense, 'N/A')
    max_clean_acc_params = response_data['model_with_defence'][max_clean_acc_defense].get('params', 'N/A')  # שינוי רביעי - תיקון הנתיב

    # Find the defense with the lowest ASR but clean accuracy within 5% of the baseline for "Balanced"
    balanced_defense = None
    balanced_asr_value = None
    balanced_clean_acc = None
    balanced_params = None

    for defense, asr in defenses_avg_ASR_wb.items():
        clean_acc = defenses_clean_acc.get(defense, None)
        if clean_acc is not None and (baseline_clean_accuracy - clean_acc) <= 0.05:
            if balanced_defense is None or asr < balanced_asr_value:
                balanced_defense = defense
                balanced_asr_value = asr
                balanced_clean_acc = clean_acc
                balanced_params = response_data['model_with_defence'][defense].get('params', 'N/A')  # שינוי חמישי - תיקון הנתיב

    # Create the table data
    data = {
        'Scenario': ['WB', '', ''],
        'Profile': ['Security-First', 'Performance-First', 'Balanced-5%'],
        'Defence': [min_asr_defense, max_clean_acc_defense, balanced_defense],
        'ASR': [min_asr_value, max_clean_acc_asr, balanced_asr_value],
        'Clean-ACC': [min_asr_clean_acc, max_clean_acc_value, balanced_clean_acc],
        'Hyperparameters': [min_asr_params, max_clean_acc_params, balanced_params]
    }

    # Convert to DataFrame for easy visualization
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size of the figure as needed

    # Create the table using the DataFrame (df)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Set font sizes and style for the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Manually set the background colors for alternating rows
    for i in range(len(df)):
        for j in range(len(df.columns)):  # Apply color for all columns
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#E5F1FB')  # Alternating row colors

    # Make the column headers bold and set background color
    for j in range(len(df.columns)):
        table[0, j].set_text_props(weight='bold')
        table[0, j].set_facecolor('#B7D4F5')

    # Manually adjust the position of the table
    table.scale(1, 2)  # Adjust scaling to fit the table properly
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Auto-adjust column widths

    # Manually adjust the position of the table within the figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Remove axes and display the table
    ax.set_axis_off()
    plt.savefig(path)
    return path


def plot_BB_table_weak_adversary(response_data, attack_mapping, strong_adv=False):
    defenses_avg_ASR_bb = {}
    defenses_clean_acc = {}
    baseline_clean_accuracy = response_data['model_without_defence']['without_attack'].get('accuracy', None)

    if strong_adv:
        section = 'optimize_attacks_on_defense_reports'
        folder = 'Strong Adversary'
    else:
        section = 'with_attack'  # שינוי ראשון - שינוי החלק הרלוונטי לפי המבנה החדש
        folder = 'Weak Adversary'

    dir = os.path.join(os.path.dirname(__file__), folder)
    path = os.path.join(dir, 'BB_Table.png')
    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Loop through the JSON to compute average ASR for BB attacks and Clean Accuracy
    for defense_name, defense_data in response_data.get('model_with_defence', {}).items():
        bb_attack_rates = []

        # Collect ASR for BB attacks
        for attack_name, attack_data in defense_data.get(section, {}).items():
            attack_success_rate = attack_data.get('attack_success_rate')  # שינוי שני - הוספת get
            if attack_success_rate is not None and attack_name in attack_mapping.get('attacks', {}) and attack_mapping['attacks'][attack_name]['assumption'] == 'BB':
                bb_attack_rates.append(attack_success_rate)

        # Calculate the average for BB if there are any BB attacks
        if bb_attack_rates:
            defenses_avg_ASR_bb[defense_name] = sum(bb_attack_rates) / len(bb_attack_rates)

        # Collect clean accuracy for each defense
        clean_acc = defense_data['without_attack'].get('accuracy', None)
        if clean_acc is not None:
            defenses_clean_acc[defense_name] = clean_acc

    if len(defenses_avg_ASR_bb) == 0:  # The dictionary is empty
        # Create a plot with the message "No Black Box Attacks"
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No Black Box Attacks', fontsize=24, ha='center', va='center', color='black')
        ax.set_axis_off()  # Hide the axes
        plt.savefig(path)  # Save the plot as an image
        plt.close()
        return None

    # Find the defense with the lowest ASR for "Security-First" (BB attacks)
    min_asr_defense_bb = min(defenses_avg_ASR_bb, key=defenses_avg_ASR_bb.get)
    min_asr_value_bb = defenses_avg_ASR_bb[min_asr_defense_bb]
    min_asr_clean_acc_bb = defenses_clean_acc.get(min_asr_defense_bb, 'N/A')
    min_asr_params_bb = response_data['model_with_defence'][min_asr_defense_bb].get('params', 'N/A')  # שינוי שלישי - תיקון הנתיב

    # Find the defense with the highest clean accuracy for "Performance-First"
    max_clean_acc_defense_bb = max(defenses_clean_acc, key=defenses_clean_acc.get)
    max_clean_acc_value_bb = defenses_clean_acc[max_clean_acc_defense_bb]
    max_clean_acc_asr_bb = defenses_avg_ASR_bb.get(max_clean_acc_defense_bb, 'N/A')
    max_clean_acc_params_bb = response_data['model_with_defence'][max_clean_acc_defense_bb].get('params', 'N/A')  # שינוי רביעי - תיקון הנתיב

    # Find the defense with the lowest ASR but clean accuracy within 5% of the baseline for "Balanced"
    balanced_defense_bb = None
    balanced_asr_value_bb = None
    balanced_clean_acc_bb = None
    balanced_params_bb = None

    for defense, asr in defenses_avg_ASR_bb.items():
        clean_acc = defenses_clean_acc.get(defense, None)
        if clean_acc is not None and (baseline_clean_accuracy - clean_acc) <= 0.05:
            if balanced_defense_bb is None or asr < balanced_asr_value_bb:
                balanced_defense_bb = defense
                balanced_asr_value_bb = asr
                balanced_clean_acc_bb = clean_acc
                balanced_params_bb = response_data['model_with_defence'][defense].get('params', 'N/A')  # שינוי חמישי - תיקון הנתיב

    # Create the table data for BB attacks
    data_bb = {
        'Scenario': ['BB', '', ''],
        'Profile': ['Security-First', 'Performance-First', 'Balanced-5%'],
        'Defence': [min_asr_defense_bb, max_clean_acc_defense_bb, balanced_defense_bb],
        'ASR': [min_asr_value_bb, max_clean_acc_asr_bb, balanced_asr_value_bb],
        'Clean-ACC': [min_asr_clean_acc_bb, max_clean_acc_value_bb, balanced_clean_acc_bb],
        'Hyperparameters': [min_asr_params_bb, max_clean_acc_params_bb, balanced_params_bb]
    }

    # Convert to DataFrame for easy visualization
    df_bb = pd.DataFrame(data_bb)

    # Create a new figure to plot the BB attack table
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size of the figure as needed

    # Create the table using the DataFrame (df_bb)
    table = ax.table(cellText=df_bb.values, colLabels=df_bb.columns, cellLoc='center', loc='center')

    # Set font sizes and style for the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Manually set the background colors for alternating rows
    for i in range(len(df_bb)):
        for j in range(len(df_bb.columns)):  # Apply color for all columns
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#E5F1FB')  # Alternating row colors

    # Make the column headers bold and set background color
    for j in range(len(df_bb.columns)):
        table[0, j].set_text_props(weight='bold')
        table[0, j].set_facecolor('#B7D4F5')

    # Manually adjust the position of the table
    table.scale(1, 2)  # Adjust scaling to fit the table properly
    table.auto_set_column_width(col=list(range(len(df_bb.columns))))  # Auto-adjust column widths

    # Manually adjust the position of the table within the figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Remove axes and display the table
    ax.set_axis_off()  # Hide the axis

    plt.savefig(path)
    return path

def plot_Clean_task_performance_table(data):
    # Extract clean task performance data
    clean_task_performance = data['model_without_defence'].get('without_attack', {})

    # Create a table of performance metrics
    # metrics = ['Accuracy', 'TPR', 'FPR', 'Precision', 'Recall', 'f1score', 'AUC']
    metrics = ['Accuracy', 'Precision', 'Recall', 'f1score', 'AUC']
    performance_data = [(metric, clean_task_performance.get(metric.lower(), 'N/A')) for metric in metrics]

    # Convert to a DataFrame for easier visualization
    df = pd.DataFrame(performance_data, columns=['Metric', 'Value'])

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the height to match the table

    # Hide the axes
    ax.set_axis_off()

    # Create the table without rowColors and manually position it
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Set font sizes and style for the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Manually set the background colors for alternating rows
    for i in range(len(df)):
        if i % 2 == 0:
            table[(i + 1, 0)].set_facecolor('#E5F1FB')  # Color the 'Metric' column
            table[(i + 1, 1)].set_facecolor('#E5F1FB')  # Color the 'Value' column

    # Make the column headers bold and set background color
    table[0, 0].set_text_props(weight='bold')
    table[0, 1].set_text_props(weight='bold')
    table[0, 0].set_facecolor('#B7D4F5')
    table[0, 1].set_facecolor('#B7D4F5')

    # Manually adjust the position of the table
    table.scale(1, 2)  # Scale the table (adjust to fit properly)
    table.auto_set_column_width([0, 1])  # Auto-adjust column widths

    # Manually adjust the position of the table within the figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save the table as an image
    combined_plot_dir = os.path.join(os.path.dirname(__file__), 'Clean_Model_Performance')
    combined_plot_path = os.path.join(combined_plot_dir, 'Clean_Model_Performance.png')

    # Ensure the directory exists
    os.makedirs(combined_plot_dir, exist_ok=True)

    # Save the plot
    plt.savefig(combined_plot_path)
    plt.close()
    return combined_plot_path
