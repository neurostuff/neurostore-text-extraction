"""Helper functions for processing annotations."""


def _get_task_name(rows):
    """Extract task names from annotation rows, replacing None/Unsure with 'n/a'."""
    # Add TaskName, replacing 'None' and 'Unsure' with 'n/a'
    rows = rows[rows.label_name == 'TaskName']
    task_names = []
    for _, row in rows.iterrows():
        if row['None'] or row['Unsure']:
            task_names.append('n/a')
        else:
            task_names.append(row['selected_text'])
    return task_names


def get_annotation_summary(annotations, id_col='pmcid'):
    """Convert annotations DataFrame to a structured summary dictionary."""
    # Convert to comparable dictionary
    annotations_summary = {}
    for _id, df in annotations.groupby(id_col):
        HasRestingState = 'DesignType-RestingState' in df.label_name.values

        # Extract exclude labels
        exclude_labels = [
            label.split('-', 1)[1] 
            for label in df.label_name 
            if label.startswith('Exclude')
        ] or None

        # Extract modality labels
        modality_labels = [
            label.split('-', 1)[1] 
            for label in df.label_name 
            if label.startswith('Modality')
        ] or None

        s = {
            'pmcid': _id,
            'HasRestingState': HasRestingState,
            'annotator_name': df.annotator_name.iloc[0],
            'Exclude': exclude_labels,
            'Modality': modality_labels,
        }

        df_abstract = df[df.section == 'abstract']
        abstract_tasks = _get_task_name(df_abstract)

        df_body = df[df.section == 'body']
        body_tasks = _get_task_name(df_body)

        # Use body tasks if available, otherwise use abstract tasks
        s['TaskName'] = body_tasks or abstract_tasks

        for k in ['TaskDescription', 'Condition', 'ContrastDefinition']:
            s[k] = df_body.loc[df_body.label_name == k, 'selected_text'].tolist() or None

        annotations_summary[_id] = s

    return annotations_summary
