import math

def PET_solver(AV_state, BV_list):
    
    '''
    Calculate the PETs for the given AV_state and each vehicle in BV_list, and output the minimum PET.
    AV_state: [x, y, vx, vy, heading, yaw_rate]
    BV_list: List of vehicles, each vehicle is represented as [x, y, vx, vy, heading, yaw_rate]
    Returns: Minimum PET value (float)
    '''
    
    pet_list = []
    car_length = 5.0  

    
    for i in range(len(BV_list)):
        if BV_list[i] is None:
            continue
            
        # Lead vehicle (same lane, ahead)
        if i == 0:
            if AV_state[2] <= BV_list[i][2]:
                continue

            dist_to_conflict = (BV_list[i][0] - car_length) - AV_state[0]
            if dist_to_conflict <= 0:
                continue
            time_av = dist_to_conflict / (AV_state[2] - BV_list[i][2] + 1e-6)

            time_lead_exit = car_length / BV_list[i][2]
            pet = time_av - time_lead_exit
            if pet < 0:  
                pet = 0.0
            pet_list.append(min(pet, 5.0))
        
        # Foll vehicle (same lane, behind)
        elif i == 1:
            if AV_state[2] >= BV_list[i][2]:
                continue
            
            dist_to_conflict = AV_state[0] - (BV_list[i][0] + car_length)
            if dist_to_conflict <= 0:
                continue
            time_foll = dist_to_conflict / (BV_list[i][2] - AV_state[2] + 1e-6)

            time_av_exit = car_length / AV_state[2]
            pet = time_foll - time_av_exit
            if pet < 0: 
                pet = 0.0
            pet_list.append(min(pet, 5.0))
        
        # Cross-lane vehicles (LeftLead, LeftFoll, RightLead, RightFoll)
        else:

            try:
                k_av = math.tan(math.radians(AV_state[4] - 89.9999999999))
                k_bv = math.tan(math.radians(BV_list[i][3] - 89.9999999999))
                
                if abs(k_av - k_bv) < 1e-6:
                    continue
                    
                cross_x = (BV_list[i][1] - AV_state[1] + k_av * AV_state[0] - k_bv * BV_list[i][0]) / (k_av - k_bv)
                cross_y = AV_state[1] + k_av * (cross_x - AV_state[0])
                
                if cross_y < 40.0 or cross_y > 52.0:
                    continue
                    
                dx_av = cross_x - AV_state[0]
                dy_av = cross_y - AV_state[1]
                dist_av = math.sqrt(dx_av**2 + dy_av**2)
                speed_av = math.sqrt(AV_state[2]**2 + AV_state[3]**2)
                time_av = dist_av / (speed_av + 1e-6)
                
                dx_bv = cross_x - BV_list[i][0]
                dy_bv = cross_y - BV_list[i][1]
                dist_bv = math.sqrt(dx_bv**2 + dy_bv**2)

                speed_bv = abs(BV_list[i][2])
                time_bv = dist_bv / (speed_bv + 1e-6)
                
                time_av_head = time_av
                time_bv_tail = time_bv + car_length / (speed_bv + 1e-6)
                pet1 = time_av_head - time_bv_tail
                
                time_av_tail = time_av + car_length / (speed_av + 1e-6)
                time_bv_head = time_bv
                pet2 = time_av_tail - time_bv_head
                pet = min(abs(pet1), abs(pet2))
                pet_list.append(min(pet, 5.0))
                
            except (ValueError, ZeroDivisionError):
                continue
    
    if not pet_list:
        return 5.0
        
    min_pet = min(pet_list)
    return min_pet    
                


