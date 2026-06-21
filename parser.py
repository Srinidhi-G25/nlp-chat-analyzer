import re
import pandas as pd

# The stable, clean regex for standard messages
pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[aApP][mM]) - (.*?): (.*)'

# A secondary check just to identify system timestamps
system_msg_pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[aApP][mM])'

def parse_and_anonymize_chat(file_path):
    parsed_data = []
    name_map = {}
    user_counter = 1
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.replace('\u200e', '').replace('\u200f', '').strip() 
            if not line:
                continue 
            
            # Tier 1: Is it a normal text message?
            match = re.match(pattern, line)
            if match:
                date_time = match.group(1)
                sender_real_name = match.group(2)
                message = match.group(3)
                
                # Auto-Anonymization
                if sender_real_name not in name_map:
                    name_map[sender_real_name] = f"Person_{user_counter}"
                    user_counter += 1
                
                sender_alias = name_map[sender_real_name]
                
                # Filter media
                if "Media omitted" in message or "<Media omitted>" in message:
                    continue
                    
                parsed_data.append([date_time, sender_alias, message])
            else:
                # Tier 2: The Rejected Line Check
                # If it starts with a timestamp but has no colon, it's a system message. Skip it entirely.
                if re.match(system_msg_pattern, line):
                    continue
                
                # If it doesn't start with a timestamp, it's a genuine continuation of the previous text
                if len(parsed_data) > 0:
                    parsed_data[-1][2] += " " + line
                    
    return pd.DataFrame(parsed_data, columns=['DateTime', 'Sender', 'Message'])

# Run the updated parser
df = parse_and_anonymize_chat('raw_chat.txt') 

# Display results
print(f"Total messages successfully parsed: {len(df)}")
df.head()