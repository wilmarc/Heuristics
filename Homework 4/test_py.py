import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import timedelta
import math 



# Cargar archivo Excel

file_path = "C:/Users/wilma/OneDrive/Documentos/1. Uniandes/0. Doctorado/2. Classes/1.3 IIND 4109/Heuristics/Homework 4/generic_input_case.xlsx"

# -*- coding: utf-8 -*-
"""
Refactored Constructive Heuristic for Transport Scheduling
Focuses on day-by-day assignment prioritizing Demand and RSP compliance,
along with other constraints. Version 3: Two-Phase Daily Assignment.
"""

import pandas as pd
import numpy as np
import random
import math
import collections # Use defaultdict for easier state tracking
import traceback # For debugging errors

# --- Data Loading ---
print("--- Loading Data ---")
try:
    # IMPORTANT: Update this path to the correct location of your file
    # file_path = "generic_input_case.xlsx"
    df_horizon = pd.read_excel(file_path, sheet_name="HORIZONTE")
    df_up_data = pd.read_excel(file_path, sheet_name="BD_UP")
    df_fleet = pd.read_excel(file_path, sheet_name="FROTA")
    df_crane = pd.read_excel(file_path, sheet_name="GRUA")
    df_factory = pd.read_excel(file_path, sheet_name="FABRICA")
    df_route = pd.read_excel(file_path, sheet_name="ROTA")
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at '{file_path}'. Please update the path.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load data from Excel file. Check sheet names and file integrity.")
    print(f"Error details: {e}")
    exit()

# --- Helper Functions ---

def calculate_daily_transport_volume(vehicles, load_capacity, cycle_time):
    """Calculate the maximum volume vehicles can transport in a day."""
    if vehicles <= 0 or load_capacity <= 0 or cycle_time <= 0:
        return 0.0
    return vehicles * load_capacity * cycle_time

def check_continuity(days_list):
    """Counts continuous blocks of activity days."""
    if not days_list: return 0
    valid_days = [d for d in days_list if isinstance(d, (int, np.integer, float, np.floating)) and not pd.isna(d)]
    if not valid_days: return 0
    sorted_days = sorted(list(set(map(int, valid_days))))
    if not sorted_days: return 0
    blocks = 1
    for i in range(len(sorted_days) - 1):
        if sorted_days[i+1] != sorted_days[i] + 1:
            blocks += 1
    return blocks

# --- Constraint Validation Functions (Keep these for final check) ---
# (Copied from previous version for completeness)

# Validate daily factory demand
def validate_daily_demand(schedule_df, df_factory, flag):
    """Check if the total daily volume transported meets the factory's min/max demand."""
    if schedule_df.empty:
        factory_days = set(df_factory['DIA'].unique())
        if 'DEMANDA_MIN' not in df_factory.columns:
             print("Error: 'DEMANDA_MIN' column missing in factory data.")
             return False
        all_min_demand_zero = df_factory['DEMANDA_MIN'].le(1e-6).all()
        return all_min_demand_zero

    daily_volume = schedule_df.groupby('DIA')['VOLUME'].sum()
    merged = pd.merge(daily_volume.reset_index(), df_factory[['DIA', 'DEMANDA_MIN', 'DEMANDA_MAX']], on='DIA', how='left')
    if merged.isnull().any().any():
        print("Error: Missing factory demand data for scheduled days.")
        print(merged[merged.isnull().any(axis=1)])
        return False
    tolerance = 1e-6
    violations = merged[(merged['VOLUME'] < merged['DEMANDA_MIN'] - tolerance) | (merged['VOLUME'] > merged['DEMANDA_MAX'] + tolerance)]
    if not violations.empty:
        if flag == 1: print("Daily Demand Non-compliance:\n", violations)
        return False
    return True

# Validate daily RSP (Solid/Pulp Ratio)
def validate_daily_rsp(schedule_df, df_factory, flag):
    """Check if the daily weighted average RSP is within the factory's limits."""
    if schedule_df.empty or 'VOLUME' not in schedule_df.columns or 'RSP' not in schedule_df.columns or schedule_df['VOLUME'].sum() < 1e-6:
        return True
    schedule_df_copy = schedule_df.copy()
    schedule_df_copy['RSP'] = pd.to_numeric(schedule_df_copy['RSP'], errors='coerce')
    schedule_df_copy['VOLUME'] = pd.to_numeric(schedule_df_copy['VOLUME'], errors='coerce')
    schedule_df_copy.dropna(subset=['RSP', 'VOLUME'], inplace=True)
    if schedule_df_copy.empty: return True
    schedule_df_copy['RSP_WEIGHTED_SUM'] = schedule_df_copy['RSP'] * schedule_df_copy['VOLUME']
    daily_rsp_sum = schedule_df_copy.groupby('DIA')['RSP_WEIGHTED_SUM'].sum()
    daily_volume_sum = schedule_df_copy.groupby('DIA')['VOLUME'].sum()
    daily_avg_rsp = (daily_rsp_sum / daily_volume_sum.replace(0, np.nan))
    daily_avg_rsp = daily_avg_rsp.reset_index()
    daily_avg_rsp.rename(columns={0: 'RSP_AVG'}, inplace=True)
    merged = pd.merge(daily_avg_rsp, df_factory[['DIA', 'RSP_MIN', 'RSP_MAX']], on='DIA', how='left')
    if merged[['RSP_MIN', 'RSP_MAX']].isnull().any().any():
        print("Error: Missing factory RSP data for scheduled days with volume.")
        print(merged[merged[['RSP_MIN', 'RSP_MAX']].isnull().any(axis=1)])
        return False
    merged_valid = merged.dropna(subset=['RSP_AVG'])
    if merged_valid.empty: return True
    tolerance = 1e-6
    violations = merged_valid[(merged_valid['RSP_AVG'] < merged_valid['RSP_MIN'] - tolerance) |
                              (merged_valid['RSP_AVG'] > merged_valid['RSP_MAX'] + tolerance)]
    if not violations.empty:
        if flag == 1: print("Daily RSP non-compliance:\n", violations)
        return False
    return True

# Validate transporter fleet size usage
def validate_transporter_fleet_size(schedule_df, df_fleet):
    """Check if the total daily vehicles used by each transporter are within their min/max limits."""
    if schedule_df.empty: return True
    daily_fleet_usage = schedule_df.groupby(['DIA', 'TRANSPORTADOR'])['QTD_VEICULOS'].sum().reset_index()
    merged = pd.merge(daily_fleet_usage, df_fleet[['TRANSPORTADOR', 'FROTA_MIN', 'FROTA_MAX']], on='TRANSPORTADOR', how='left')
    if merged.isnull().any().any():
        print("Error: Missing fleet min/max data for scheduled transporters.")
        print(merged[merged.isnull().any(axis=1)])
        return False
    tolerance = 1e-6
    violations = merged[(merged['QTD_VEICULOS'] < merged['FROTA_MIN'] - tolerance) | (merged['QTD_VEICULOS'] > merged['FROTA_MAX'] + tolerance)]
    if not violations.empty:
        print("Violation - Transporter Fleet Size:\n", violations)
        return False
    return True

# Validate crane capacity usage
def validate_crane_capacity(schedule_df, df_crane):
    """Check if a transporter serves more simultaneous UPs than their available cranes on any day."""
    if schedule_df.empty: return True
    active_schedule = schedule_df[schedule_df['VOLUME'] > 1e-6]
    if active_schedule.empty: return True
    daily_up_count = active_schedule.groupby(['DIA', 'TRANSPORTADOR'])['UP'].nunique().reset_index()
    daily_up_count.rename(columns={'UP': 'UP_COUNT'}, inplace=True)
    merged = pd.merge(daily_up_count, df_crane[['TRANSPORTADOR', 'QTD_GRUAS']], on='TRANSPORTADOR', how='left')
    if merged.isnull().any().any():
        print("Error: Missing crane quantity data for scheduled transporters.")
        print(merged[merged.isnull().any(axis=1)])
        return False
    merged['QTD_GRUAS'] = merged['QTD_GRUAS'].fillna(0)
    violations = merged[merged['UP_COUNT'] > merged['QTD_GRUAS']]
    if not violations.empty:
        print("Violation - Crane Capacity (Serving too many UPs simultaneously):\n", violations)
        return False
    return True

# Validate Fazenda (Farm) exclusivity
def validate_fazenda_exclusivity(schedule_df):
    """Check if a transporter operates in more than one Fazenda on the same day."""
    if schedule_df.empty: return True
    active_schedule = schedule_df[schedule_df['VOLUME'] > 1e-6]
    if active_schedule.empty: return True
    daily_fazenda_count = active_schedule.groupby(['DIA', 'TRANSPORTADOR'])['FAZENDA'].nunique().reset_index()
    daily_fazenda_count.rename(columns={'FAZENDA': 'FAZENDA_COUNT'}, inplace=True)
    violations = daily_fazenda_count[daily_fazenda_count['FAZENDA_COUNT'] > 1]
    if not violations.empty:
        print("Violation - Fazenda Exclusivity (Transporter in multiple Fazendas on the same day):\n", violations)
        return False
    return True

# Validate minimum percentage of vehicles assigned per UP
def validate_min_vehicles_per_up(schedule_df, df_crane):
    """Check if the minimum percentage of a transporter's active vehicles assigned to a single UP is met."""
    if schedule_df.empty: return True
    total_daily_fleet = schedule_df.groupby(['DIA', 'TRANSPORTADOR'])['QTD_VEICULOS'].sum().reset_index()
    total_daily_fleet.rename(columns={'QTD_VEICULOS': 'TOTAL_VEHICLES_DAY'}, inplace=True)
    merged = pd.merge(schedule_df, total_daily_fleet, on=['DIA', 'TRANSPORTADOR'], how='left')
    if 'PORCENTAGEM_VEICULOS_MIN' not in df_crane.columns:
        print("Error: 'PORCENTAGEM_VEICULOS_MIN' column missing in crane data.")
        return False
    merged = pd.merge(merged, df_crane[['TRANSPORTADOR', 'PORCENTAGEM_VEICULOS_MIN']], on='TRANSPORTADOR', how='left')
    if merged['TOTAL_VEHICLES_DAY'].isnull().any() or merged['PORCENTAGEM_VEICULOS_MIN'].isnull().any():
        print("Error: Missing total vehicle or min percentage data during validation.")
        print(merged[merged.isnull().any(axis=1)])
        return False
    merged['ACTUAL_PERCENTAGE'] = 0.0
    non_zero_fleet_mask = merged['TOTAL_VEHICLES_DAY'] > 1e-9
    merged.loc[non_zero_fleet_mask, 'ACTUAL_PERCENTAGE'] = merged['QTD_VEICULOS'] / merged['TOTAL_VEHICLES_DAY']
    tolerance = 1e-6
    violations = merged[non_zero_fleet_mask & (merged['QTD_VEICULOS'] > 1e-9) &
                        (merged['ACTUAL_PERCENTAGE'] < merged['PORCENTAGEM_VEICULOS_MIN'] - tolerance)]
    if not violations.empty:
        print("Violation - Minimum Vehicles per UP Percentage:\n", violations[['DIA', 'TRANSPORTADOR', 'UP', 'QTD_VEICULOS', 'TOTAL_VEHICLES_DAY', 'PORCENTAGEM_VEICULOS_MIN', 'ACTUAL_PERCENTAGE']])
        return False
    return True

# Validate UP Completion Rules
def validate_completion_rules(schedule_df, df_up_data):
    """Validate the transport completion rules for UPs based on their total volume."""
    all_valid = True
    if schedule_df is None or schedule_df.empty: return True
    total_volume_transported = schedule_df.groupby('UP')['VOLUME'].sum()
    scheduled_ups = schedule_df['UP'].unique()
    for up_id in scheduled_ups:
        try:
            up_info = df_up_data[df_up_data['UP'] == up_id]
            if up_info.empty:
                print(f"Warning: UP {up_id} found in schedule but not in UP data. Skipping completion check.")
                continue
            up_info = up_info.iloc[0]
            total_volume_up = up_info['VOLUME']
            volume_transported_up = total_volume_transported.get(up_id, 0)
            up_schedule = schedule_df[schedule_df['UP'] == up_id]
            days_worked = up_schedule[up_schedule['VOLUME'] > 1e-6]['DIA'].tolist()
            num_blocks = check_continuity(days_worked)
            is_small_up = total_volume_up < 7000
            if is_small_up:
                if 1e-6 < volume_transported_up < total_volume_up - 1e-6:
                     print(f"Violation UP < 7000: UP {up_id} started but not completed. Transported: {volume_transported_up:.2f}, Total: {total_volume_up:.2f}")
                     all_valid = False
                if volume_transported_up > 1e-6 and num_blocks > 1:
                     print(f"Violation UP < 7000: UP {up_id} has {num_blocks} entries (must be 1). Days: {sorted(list(set(map(int, filter(lambda x: isinstance(x, (int, float)) and not pd.isna(x), days_worked)))))}")
                     all_valid = False
            else: # Large UP
                if volume_transported_up > 1e-6 and num_blocks > 2:
                     print(f"Violation UP >= 7000: UP {up_id} has {num_blocks} entries (max 2 allowed). Days: {sorted(list(set(map(int, filter(lambda x: isinstance(x, (int, float)) and not pd.isna(x), days_worked)))))}")
                     all_valid = False
        except Exception as e:
            print(f"Error during completion rule validation for UP {up_id}: {e}")
            all_valid = False
            continue
    return all_valid

# Main Validation Function (All Constraints)
def validate_all_constraints(schedule_df, df_factory, df_fleet, df_crane, df_up_data, flag):
    """Runs all validation checks including the complex completion rules."""
    if flag == 1: print("--- Starting Full Constraint Validation --- ")
    is_feasible = True
    schedule_df_copy = schedule_df.copy()
    if not validate_daily_demand(schedule_df_copy, df_factory, flag): is_feasible = False
    if not validate_daily_rsp(schedule_df_copy, df_factory, flag): is_feasible = False
    if not validate_transporter_fleet_size(schedule_df_copy, df_fleet): is_feasible = False
    if not validate_crane_capacity(schedule_df_copy, df_crane): is_feasible = False
    if not validate_fazenda_exclusivity(schedule_df_copy): is_feasible = False
    # if not validate_min_vehicles_per_up(schedule_df_copy, df_crane): is_feasible = False # Optional
    if not validate_completion_rules(schedule_df_copy, df_up_data): is_feasible = False
    if flag == 1:
        print("--- Validation Finished --- ")
        if is_feasible: print(">>> Solution COMPLIES with all verified constraints.")
        else: print(">>> Solution DOES NOT COMPLY with one or more constraints (see details above).")
    return is_feasible
# ---------------------------------------------------------------------------


# --- Main Constructive Heuristic Function (Demand/RSP Focus v3 - Two Phase) ---

def run_constructive_heuristic(df_horizon, df_up_data, df_fleet, df_crane, df_factory, df_route):
    """
    Constructs a schedule day by day using a two-phase approach within each day
    to prioritize Demand and RSP compliance.
    """
    print("--- Initializing Heuristic (Demand/RSP Focused v3 - Two Phase) ---")

    # --- Preprocessing and Lookups ---
    days = df_horizon['DIA'].tolist()
    ups_all = df_up_data['UP'].tolist()
    up_data_dict = df_up_data.set_index('UP').to_dict('index')
    transporters = df_fleet['TRANSPORTADOR'].unique().tolist()
    fleet_dict = df_fleet.set_index('TRANSPORTADOR').to_dict('index')
    crane_dict = df_crane.set_index('TRANSPORTADOR').to_dict('index')
    factory_demand_dict = df_factory.set_index('DIA').to_dict('index')
    route_dict = {}
    for _, row in df_route.iterrows():
        route_dict[(row['ORIGEM'], row['TRANSPORTADOR'])] = {
            'load_capacity': row['CAIXA_CARGA'], 'cycle_time': row['TEMPO_CICLO']
        }
    transp_per_up = collections.defaultdict(list)
    for (up, t), _ in route_dict.items():
        if t not in transp_per_up[up]: transp_per_up[up].append(t)
    ups_in_fazenda = collections.defaultdict(list)
    for up, data in up_data_dict.items():
        ups_in_fazenda[data['FAZENDA']].append(up)

    # --- UP Prioritization ---
    print("--- Prioritizing UPs ---")
    try:
        rsp_min_global = df_factory['RSP_MIN'].min()
        rsp_max_global = df_factory['RSP_MAX'].max()
    except (KeyError, IndexError):
        rsp_min_global, rsp_max_global = 0, float('inf')

    df_up_data['IS_RSP_COMPLIANT_GLOBAL'] = (df_up_data['RSP'] >= rsp_min_global) & (df_up_data['RSP'] <= rsp_max_global)
    df_up_sorted = df_up_data.sort_values(
        by=['IS_RSP_COMPLIANT_GLOBAL', 'VOLUME'], ascending=[False, False]
    ).reset_index(drop=True)
    prioritized_up_list_base = df_up_sorted['UP'].tolist()
    print(f"Base UP Prioritization Order (Top 5): {prioritized_up_list_base[:5]}")

    # --- State Tracking Initialization ---
    up_status = {}
    for up in ups_all:
        up_status[up] = {
            'remaining_volume': up_data_dict[up]['VOLUME'], 'days_worked': [],
            'work_blocks': 0, 'status': 'PENDING',
            'total_volume': up_data_dict[up]['VOLUME'], 'fazenda': up_data_dict[up]['FAZENDA'],
            'rsp': up_data_dict[up]['RSP'], 'db': up_data_dict[up]['DB']
        }

    schedule_assignments = [] # Stores confirmed assignment dicts

    # --- Main Daily Loop ---
    print("--- Starting Daily Assignment Loop ---")
    for d in days:
        print(f"\n===== Processing Day {d} =====")

        # Get daily limits
        factory_day_info = factory_demand_dict.get(d, {})
        demand_min_today = factory_day_info.get('DEMANDA_MIN', 0)
        demand_max_today = factory_day_info.get('DEMANDA_MAX', float('inf'))
        rsp_min_today = factory_day_info.get('RSP_MIN', 0)
        rsp_max_today = factory_day_info.get('RSP_MAX', float('inf'))

        # Daily Transporter Status Reset
        transporter_day_status = {}
        for t in transporters:
            transporter_day_status[t] = {
                'vehicles_assigned': 0, 'ups_assigned': [], 'fazenda_assigned': None
            }

        # Track assignments made specifically for this day
        assignments_today = []

        # --- PHASE 1: Meet Minimum Demand ---
        print(f"  --- Day {d}: Phase 1 - Meet Min Demand ({demand_min_today:.2f}) ---")
        MAX_DAILY_ITERATIONS_PHASE1 = 200 # Limit iterations
        for daily_iter in range(MAX_DAILY_ITERATIONS_PHASE1):

            current_day_volume = sum(a['VOLUME'] for a in assignments_today)
            current_day_rsp_sum = sum(a['RSP'] * a['VOLUME'] for a in assignments_today)
            current_avg_rsp = (current_day_rsp_sum / current_day_volume) if current_day_volume > 1e-6 else None
            rsp_in_range = (current_avg_rsp is not None and rsp_min_today - 1e-6 <= current_avg_rsp <= rsp_max_today + 1e-6)
            if current_avg_rsp is None: rsp_in_range = True # Treat as OK if no volume yet

            demand_min_met = current_day_volume >= demand_min_today - 1e-6
            if demand_min_met:
                print(f"  Day {d}, Iter {daily_iter}: Min demand met ({current_day_volume:.2f}). Moving to Phase 2.")
                break # Exit Phase 1

            possible_assignments_p1 = []
            available_ups = [up for up, state in up_status.items() if state['status'] != 'COMPLETED' and state['remaining_volume'] > 1e-6]

            for up in available_ups:
                # ... (Constraint checks: Transporter availability, Fazenda, Crane, UP Completion) ...
                # (Same checks as before to find valid UP-Transporter pairs)
                up_state = up_status[up]
                possible_transporters = transp_per_up.get(up, [])
                available_transporters = []
                for t in possible_transporters:
                    t_status = transporter_day_status[t]
                    t_fleet_info = fleet_dict.get(t, {})
                    t_max_fleet = t_fleet_info.get('FROTA_MAX', 0)
                    if t_status['vehicles_assigned'] >= t_max_fleet: continue
                    current_fazenda = up_state['fazenda']
                    if t_status['fazenda_assigned'] is not None and t_status['fazenda_assigned'] != current_fazenda: continue
                    t_crane_info = crane_dict.get(t, {})
                    t_max_cranes = t_crane_info.get('QTD_GRUAS', 0)
                    ups_in_fazenda_today = [u for u in t_status['ups_assigned'] if up_status[u]['fazenda'] == current_fazenda]
                    if len(ups_in_fazenda_today) >= t_max_cranes: continue
                    days_worked_list = up_state['days_worked']
                    is_new_block_today = (not days_worked_list) or (d != days_worked_list[-1] + 1)
                    potential_blocks_if_worked = up_state['work_blocks']
                    if is_new_block_today and days_worked_list: potential_blocks_if_worked += 1
                    elif not days_worked_list: potential_blocks_if_worked = 1
                    can_work_on_up = False
                    if up_state['total_volume'] < 7000:
                        if potential_blocks_if_worked <= 1: can_work_on_up = True
                    else:
                        if potential_blocks_if_worked <= 2: can_work_on_up = True
                    if not can_work_on_up: continue
                    available_transporters.append(t)
                #-------------------------------------------------------------------

                for t in available_transporters:
                    # Calculate potential assignment volume & vehicles
                    route_info = route_dict.get((up, t))
                    if not route_info: continue
                    load_cap, cycle_t = route_info['load_capacity'], route_info['cycle_time']
                    if load_cap <= 0 or cycle_t <= 0: continue

                    t_status = transporter_day_status[t]
                    t_fleet_info = fleet_dict.get(t, {})
                    t_max_fleet = t_fleet_info.get('FROTA_MAX', 0)
                    vehicles_remaining_for_t = t_max_fleet - t_status['vehicles_assigned']
                    remaining_volume_up = up_state['remaining_volume']
                    daily_cap_per_vehicle = load_cap * cycle_t

                    # Estimate vehicles needed to reach min demand or clear UP
                    volume_needed = max(0, demand_min_today - current_day_volume)
                    target_vol_for_calc = min(remaining_volume_up, volume_needed if volume_needed > 1e-6 else remaining_volume_up)
                    vehicles_needed_approx = 0
                    if daily_cap_per_vehicle > 0:
                        vehicles_needed_approx = math.ceil(target_vol_for_calc / daily_cap_per_vehicle)

                    assign_veh = min(vehicles_remaining_for_t, vehicles_needed_approx if vehicles_needed_approx > 0 else 1)
                    assign_veh = max(1, assign_veh) # Try at least 1
                    assign_veh = min(assign_veh, vehicles_remaining_for_t) # Ensure not exceeding available
                    if assign_veh <= 0: continue

                    vol_potencial = calculate_daily_transport_volume(assign_veh, load_cap, cycle_t)
                    actual_volume = 0
                    if up_state['total_volume'] < 7000: # Small UP Rule
                        if vol_potencial >= remaining_volume_up - 1e-6: actual_volume = remaining_volume_up
                        else: actual_volume = 0
                    else: # Large UP Rule
                        actual_volume = min(vol_potencial, remaining_volume_up)
                    actual_volume = max(0, actual_volume)

                    # Adjust if exceeds max demand (shouldn't happen often in Phase 1, but safety)
                    volume_headroom = demand_max_today - current_day_volume
                    if actual_volume > volume_headroom + 1e-6:
                         actual_volume = max(0, volume_headroom)
                         if daily_cap_per_vehicle > 0:
                              assign_veh_capped = math.ceil(actual_volume / daily_cap_per_vehicle)
                              assign_veh = min(assign_veh, assign_veh_capped)
                         else: assign_veh = 0; actual_volume = 0

                    if actual_volume < 1e-6 or assign_veh <= 0: continue

                    # --- Score Assignment for Phase 1 ---
                    score = 0
                    up_rsp = up_state['rsp']
                    volume_added = actual_volume

                    score += 1000 * volume_added # Primary: Add volume

                    # Secondary: Help RSP if needed
                    if current_avg_rsp is not None and not rsp_in_range:
                        if current_avg_rsp < rsp_min_today and up_rsp > current_avg_rsp: score += 100 * (up_rsp - current_avg_rsp)
                        elif current_avg_rsp > rsp_max_today and up_rsp < current_avg_rsp: score += 100 * (current_avg_rsp - up_rsp)

                    # Tie-breaker: Base priority
                    try: score -= prioritized_up_list_base.index(up) * 0.01
                    except ValueError: pass

                    possible_assignments_p1.append({'score': score, 'up': up, 't': t,
                                                 'assign_veh': assign_veh, 'actual_volume': actual_volume,
                                                 'up_rsp': up_rsp, 'up_db': up_state['db']})

            # --- Select and Make Best Assignment for Phase 1 ---
            if not possible_assignments_p1:
                print(f"  Day {d}, Iter {daily_iter}: No more assignments possible in Phase 1.")
                break # Exit Phase 1 loop

            best_assignment = max(possible_assignments_p1, key=lambda x: x['score'])

            # Perform assignment (update state, add to assignments_today)
            up = best_assignment['up']; t = best_assignment['t']
            assign_veh = best_assignment['assign_veh']; actual_volume = best_assignment['actual_volume']
            up_state = up_status[up]; t_status = transporter_day_status[t]
            current_fazenda = up_state['fazenda']
            print(f"  Phase 1 Assign (Iter {daily_iter}, Score {best_assignment['score']:.2f}): UP {up}, T {t}, Veh {assign_veh}, Vol {actual_volume:.2f}")

            up_state['remaining_volume'] -= actual_volume
            if up_state['remaining_volume'] < 1e-6: up_state['remaining_volume'] = 0; up_state['status'] = 'COMPLETED'
            else: up_state['status'] = 'ACTIVE'
            if d not in up_state['days_worked']:
                up_state['days_worked'].append(d)
                days_worked_list = up_state['days_worked']
                is_new_block_today = (len(days_worked_list) == 1) or (d != days_worked_list[-2] + 1)
                if is_new_block_today: up_state['work_blocks'] += 1
            t_status['vehicles_assigned'] += assign_veh
            t_status['ups_assigned'].append(up)
            t_status['fazenda_assigned'] = current_fazenda
            assignment_details = { 'UP': up, 'FAZENDA': current_fazenda, 'TRANSPORTADOR': t, 'DIA': d, 'MES': df_horizon[df_horizon['DIA'] == d]['MES'].iloc[0], 'DB': best_assignment['up_db'], 'RSP': best_assignment['up_rsp'], 'QTD_VEICULOS': assign_veh, 'VOLUME': actual_volume }
            assignments_today.append(assignment_details)

        # End Phase 1 Loop
        if daily_iter == MAX_DAILY_ITERATIONS_PHASE1 - 1:
             print(f"Warning: Reached max iterations ({MAX_DAILY_ITERATIONS_PHASE1}) for day {d} in Phase 1.")


        # --- PHASE 2: Correct RSP & Reach Max Demand ---
        print(f"  --- Day {d}: Phase 2 - Correct RSP / Reach Max Demand ({demand_max_today:.2f}) ---")
        MAX_DAILY_ITERATIONS_PHASE2 = 300 # Limit iterations
        for daily_iter in range(MAX_DAILY_ITERATIONS_PHASE2):

            # Recalculate current state
            current_day_volume = sum(a['VOLUME'] for a in assignments_today)
            current_day_rsp_sum = sum(a['RSP'] * a['VOLUME'] for a in assignments_today)
            current_avg_rsp = (current_day_rsp_sum / current_day_volume) if current_day_volume > 1e-6 else None
            rsp_in_range = (current_avg_rsp is not None and rsp_min_today - 1e-6 <= current_avg_rsp <= rsp_max_today + 1e-6)
            if current_avg_rsp is None: rsp_in_range = True
            demand_max_reached = current_day_volume >= demand_max_today - 1e-6

            # Stop conditions for Phase 2
            if demand_max_reached:
                print(f"  Day {d}, Iter {daily_iter}: Max demand reached in Phase 2. Stopping.")
                break
            if rsp_in_range:
                 # If RSP is okay, maybe stop early or only add volume cautiously
                 print(f"  Day {d}, Iter {daily_iter}: RSP is in range ({current_avg_rsp:.3f}). Focusing on volume up to max.")
                 # We can continue to add volume up to max_demand if RSP is okay

            possible_assignments_p2 = []
            available_ups = [up for up, state in up_status.items() if state['status'] != 'COMPLETED' and state['remaining_volume'] > 1e-6]

            for up in available_ups:
                # ... (Constraint checks: Transporter availability, Fazenda, Crane, UP Completion) ...
                # (Same checks as in Phase 1)
                up_state = up_status[up]
                possible_transporters = transp_per_up.get(up, [])
                available_transporters = []
                for t in possible_transporters:
                    t_status = transporter_day_status[t]
                    t_fleet_info = fleet_dict.get(t, {})
                    t_max_fleet = t_fleet_info.get('FROTA_MAX', 0)
                    if t_status['vehicles_assigned'] >= t_max_fleet: continue
                    current_fazenda = up_state['fazenda']
                    if t_status['fazenda_assigned'] is not None and t_status['fazenda_assigned'] != current_fazenda: continue
                    t_crane_info = crane_dict.get(t, {})
                    t_max_cranes = t_crane_info.get('QTD_GRUAS', 0)
                    ups_in_fazenda_today = [u for u in t_status['ups_assigned'] if up_status[u]['fazenda'] == current_fazenda]
                    if len(ups_in_fazenda_today) >= t_max_cranes: continue
                    days_worked_list = up_state['days_worked']
                    is_new_block_today = (not days_worked_list) or (d != days_worked_list[-1] + 1)
                    potential_blocks_if_worked = up_state['work_blocks']
                    if is_new_block_today and days_worked_list: potential_blocks_if_worked += 1
                    elif not days_worked_list: potential_blocks_if_worked = 1
                    can_work_on_up = False
                    if up_state['total_volume'] < 7000:
                        if potential_blocks_if_worked <= 1: can_work_on_up = True
                    else:
                        if potential_blocks_if_worked <= 2: can_work_on_up = True
                    if not can_work_on_up: continue
                    available_transporters.append(t)
                #-------------------------------------------------------------------


                for t in available_transporters:
                    # Calculate potential assignment volume & vehicles
                    route_info = route_dict.get((up, t))
                    if not route_info: continue
                    load_cap, cycle_t = route_info['load_capacity'], route_info['cycle_time']
                    if load_cap <= 0 or cycle_t <= 0: continue

                    t_status = transporter_day_status[t]
                    t_fleet_info = fleet_dict.get(t, {})
                    t_max_fleet = t_fleet_info.get('FROTA_MAX', 0)
                    vehicles_remaining_for_t = t_max_fleet - t_status['vehicles_assigned']
                    remaining_volume_up = up_state['remaining_volume']
                    daily_cap_per_vehicle = load_cap * cycle_t

                    # Estimate vehicles needed (focus on remaining UP vol or filling to max demand)
                    volume_headroom = max(0, demand_max_today - current_day_volume)
                    target_vol_for_calc = min(remaining_volume_up, volume_headroom) # Target filling headroom

                    vehicles_needed_approx = 0
                    if daily_cap_per_vehicle > 0:
                        vehicles_needed_approx = math.ceil(target_vol_for_calc / daily_cap_per_vehicle)

                    assign_veh = min(vehicles_remaining_for_t, vehicles_needed_approx if vehicles_needed_approx > 0 else 1)
                    assign_veh = max(1, assign_veh)
                    assign_veh = min(assign_veh, vehicles_remaining_for_t)
                    if assign_veh <= 0: continue

                    vol_potencial = calculate_daily_transport_volume(assign_veh, load_cap, cycle_t)
                    actual_volume = 0
                    if up_state['total_volume'] < 7000: # Small UP Rule
                        if vol_potencial >= remaining_volume_up - 1e-6: actual_volume = remaining_volume_up
                        else: actual_volume = 0
                    else: # Large UP Rule
                        actual_volume = min(vol_potencial, remaining_volume_up)
                    actual_volume = max(0, actual_volume)

                    # Ensure doesn't exceed max demand
                    actual_volume = min(actual_volume, volume_headroom + 1e-6)
                    actual_volume = max(0, actual_volume) # Re-check after capping

                    # Recalculate vehicles if volume was capped significantly
                    if actual_volume < vol_potencial - 1e-6:
                         if daily_cap_per_vehicle > 0:
                              assign_veh_capped = math.ceil(actual_volume / daily_cap_per_vehicle)
                              assign_veh = min(assign_veh, assign_veh_capped)
                         else: assign_veh = 0; actual_volume = 0

                    if actual_volume < 1e-6 or assign_veh <= 0: continue

                    # --- Score Assignment for Phase 2 ---
                    score = 0
                    up_rsp = up_state['rsp']
                    volume_added = actual_volume
                    new_total_volume = current_day_volume + volume_added
                    new_rsp_sum = current_day_rsp_sum + (up_rsp * volume_added)
                    new_avg_rsp = (new_rsp_sum / new_total_volume) if new_total_volume > 1e-6 else None

                    if new_avg_rsp is not None:
                         rsp_target_met_now = (rsp_min_today - 1e-6 <= new_avg_rsp <= rsp_max_today + 1e-6)
                         # Priority: Fix RSP if it's wrong
                         if not rsp_in_range:
                             if rsp_target_met_now: score += 3000 # High reward for fixing RSP
                             else: # Check direction
                                 if current_avg_rsp < rsp_min_today and new_avg_rsp > current_avg_rsp: score += 1500 * (new_avg_rsp - current_avg_rsp)
                                 elif current_avg_rsp > rsp_max_today and new_avg_rsp < current_avg_rsp: score += 1500 * (current_avg_rsp - new_avg_rsp)
                                 else: score -= 1000 # Penalize worsening
                         # If RSP is okay, reward keeping it okay, and adding volume
                         elif rsp_target_met_now:
                              score += 500 # Reward keeping RSP ok
                              score += volume_added # Secondary: add volume towards max
                         else: # Move makes RSP bad
                              score -= 2000 # Penalize breaking RSP range

                    # Tie-breaker: Base priority
                    try: score -= prioritized_up_list_base.index(up) * 0.01
                    except ValueError: pass

                    possible_assignments_p2.append({'score': score, 'up': up, 't': t,
                                                 'assign_veh': assign_veh, 'actual_volume': actual_volume,
                                                 'up_rsp': up_rsp, 'up_db': up_state['db']})

            # --- Select and Make Best Assignment for Phase 2 ---
            if not possible_assignments_p2:
                print(f"  Day {d}, Iter {daily_iter}: No more assignments possible in Phase 2.")
                break # Exit Phase 2 loop

            best_assignment = max(possible_assignments_p2, key=lambda x: x['score'])

            # Stop if best score is too low (avoid minor volume additions if RSP ok)
            if rsp_in_range and best_assignment['score'] < 10: # Low threshold if RSP ok
                 print(f"  Day {d}, Iter {daily_iter}: Best assignment score ({best_assignment['score']:.2f}) too low in Phase 2. Stopping.")
                 break
            if best_assignment['score'] < -100: # Stop if move seems detrimental
                 print(f"  Day {d}, Iter {daily_iter}: Best assignment score ({best_assignment['score']:.2f}) is negative. Stopping.")
                 break

            # Perform assignment
            up = best_assignment['up']; t = best_assignment['t']
            assign_veh = best_assignment['assign_veh']; actual_volume = best_assignment['actual_volume']
            up_state = up_status[up]; t_status = transporter_day_status[t]
            current_fazenda = up_state['fazenda']
            print(f"  Phase 2 Assign (Iter {daily_iter}, Score {best_assignment['score']:.2f}): UP {up}, T {t}, Veh {assign_veh}, Vol {actual_volume:.2f}")

            up_state['remaining_volume'] -= actual_volume
            if up_state['remaining_volume'] < 1e-6: up_state['remaining_volume'] = 0; up_state['status'] = 'COMPLETED'
            else: up_state['status'] = 'ACTIVE'
            if d not in up_state['days_worked']:
                up_state['days_worked'].append(d)
                days_worked_list = up_state['days_worked']
                is_new_block_today = (len(days_worked_list) == 1) or (d != days_worked_list[-2] + 1)
                if is_new_block_today: up_state['work_blocks'] += 1
            t_status['vehicles_assigned'] += assign_veh
            t_status['ups_assigned'].append(up)
            t_status['fazenda_assigned'] = current_fazenda
            assignment_details = { 'UP': up, 'FAZENDA': current_fazenda, 'TRANSPORTADOR': t, 'DIA': d, 'MES': df_horizon[df_horizon['DIA'] == d]['MES'].iloc[0], 'DB': best_assignment['up_db'], 'RSP': best_assignment['up_rsp'], 'QTD_VEICULOS': assign_veh, 'VOLUME': actual_volume }
            assignments_today.append(assignment_details)

        # End Phase 2 Loop
        if daily_iter == MAX_DAILY_ITERATIONS_PHASE2 - 1:
             print(f"Warning: Reached max iterations ({MAX_DAILY_ITERATIONS_PHASE2}) for day {d} in Phase 2.")

        # Add assignments made today to the overall schedule
        schedule_assignments.extend(assignments_today)

        # --- Post-Day Check: Min Fleet Size ---
        for t in transporters:
             t_status = transporter_day_status[t]
             t_fleet_info = fleet_dict.get(t, {})
             t_min_fleet = t_fleet_info.get('FROTA_MIN', 0)
             if t_status['vehicles_assigned'] > 0 and t_status['vehicles_assigned'] < t_min_fleet:
                  print(f"Warning: Day {d}, Transporter {t} assigned {t_status['vehicles_assigned']} vehicles (Min: {t_min_fleet}).")


    # --- End Main Daily Loop ---

    print("\n--- Heuristic Finished ---")
    if not schedule_assignments:
        print("Warning: No assignments were made.")
        return pd.DataFrame()

    final_schedule_df = pd.DataFrame(schedule_assignments)
    # Ensure correct data types before sorting/saving
    final_schedule_df['DIA'] = final_schedule_df['DIA'].astype(int)
    final_schedule_df['MES'] = final_schedule_df['MES'].astype(int)
    final_schedule_df['QTD_VEICULOS'] = final_schedule_df['QTD_VEICULOS'].astype(int)
    final_schedule_df['VOLUME'] = final_schedule_df['VOLUME'].astype(float)
    final_schedule_df['DB'] = final_schedule_df['DB'].astype(float)
    final_schedule_df['RSP'] = final_schedule_df['RSP'].astype(float)

    final_schedule_df = final_schedule_df.sort_values(by=['DIA', 'TRANSPORTADOR', 'UP']).reset_index(drop=True)

    # --- Final Validation ---
    print("\n--- Running Final Validation ---")
    validate_all_constraints(final_schedule_df, df_factory, df_fleet, df_crane, df_up_data, flag=1)

    return final_schedule_df


# --- Main Execution ---
if __name__ == "__main__":
    # Run the heuristic
    final_schedule = run_constructive_heuristic(
        df_horizon, df_up_data, df_fleet, df_crane, df_factory, df_route
    )

    # Display results
    if final_schedule is not None and not final_schedule.empty:
        print("\n--- Final Generated Schedule ---")
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(final_schedule)

        # Save results
        try:
            output_filename = "heuristic_schedule_output_v4_two_phase.csv"
            # Adjust separator/decimal based on your region's Excel settings if needed
            final_schedule.to_csv(output_filename, index=False, sep=';', decimal=',')
            print(f"\nSchedule saved to {output_filename}")
        except Exception as e:
            print(f"\nError saving schedule to CSV: {e}")
    else:
        print("\nHeuristic did not produce a schedule.")
