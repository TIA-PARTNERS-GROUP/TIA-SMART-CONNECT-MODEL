import pandas as pd
import random

# Define possible capabilities and industries
capabilities = [
    "CCTV Installer", "Software Developer", "Creative Designer", "Researcher", 
    "Manufacturer", "Logistics", "Brand Strategist", "UI/UX Expert", 
    "AI Consultant", "Legal Advisor", "Financial Analyst", "Engineer"
]

industries = [
    "Tech", "Security", "Logistics", "Manufacturing", "Research", 
    "Creative", "Finance", "Legal", "Healthcare", "Construction"
]

countries = ["USA", "Germany", "India", "Brazil", "Australia", "Canada", "UK", "China"]

# Generate fake dataset
num_entries = 1000
data = []

for i in range(num_entries):
    name = f"Entity_{i+1}"
    country = random.choice(countries)
    capability = random.choice(capabilities)
    industry = random.choice(industries)
    year_founded = random.randint(1980, 2023)
    revenue = round(random.uniform(1, 1000), 2)  # in millions
    data.append({
        "Name": name,
        "Country": country,
        "Capability": capability,
        "Industry": industry,
        "Year Founded": year_founded,
        "Annual Revenue (M USD)": revenue
    })

# Convert to DataFrame
df = pd.DataFrame(data)
df.head()
