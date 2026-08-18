import re

with open('d:/MAIN PROJECT CV AND DL/HydraFusion/scripts/generate_html_report.py', 'r', encoding='utf-8') as f:
    text = f.read()

parts = text.split('<script>', 1)
if len(parts) == 2:
    html_part, script_part = parts
    
    html_part = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_part)
    html_part = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', html_part)
    
    text = html_part + '<script>' + script_part

with open('d:/MAIN PROJECT CV AND DL/HydraFusion/scripts/generate_html_report.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
