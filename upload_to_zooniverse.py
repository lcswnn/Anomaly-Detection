from panoptes_client import Panoptes, Project, SubjectSet, Subject
import pandas as pd
import os

# Connect
Panoptes.connect(username=os.environ['ZOONIVERSE_USER'], password=os.environ['ZOONIVERSE_PASS'])

# Find your project
project = Project.find(slug='lucaswaunn/astronomy-anomaly-classification')
print(f"Found project: {project.title}")

# Create subject set
subject_set = SubjectSet()
subject_set.links.project = project
subject_set.display_name = 'Anomaly Batch 1'
subject_set.save()
print(f"Subject set created with ID: {subject_set.id}")

# Load manifest
manifest = pd.read_csv('manifest.csv')
image_dir = 'zoon_imgs'

print(f"Uploading {len(manifest)} subjects...")

new_subjects = []
failed = []

for i, row in manifest.iterrows():
    filepath = os.path.join(image_dir, row['filename'])
    
    if not os.path.exists(filepath):
        failed.append(row['filename'])
        continue
    
    subject = Subject()
    subject.links.project = project
    subject.add_location(filepath)
    subject.metadata['tile_id'] = str(row['tile_id'])
    subject.metadata['mse'] = str(row['mse'])
    subject.save()
    new_subjects.append(subject)
    
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(manifest)} uploaded...")

# Link subjects to set
subject_set.add(new_subjects)

print(f"\nDone! Uploaded {len(new_subjects)} subjects")
if failed:
    print(f"Warning: {len(failed)} files not found")