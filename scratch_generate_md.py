import csv
import math
import os

def read_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def format_row(row):
    stem = row['stem']
    
    # Path formatters
    gt_path = f"dataset/braid/img/test/{stem}.png"
    sk_path = f"dataset/braid/sketch/test/{stem}.png"
    gan_path = f"custom_results/gan/shs/{stem}.png"
    dit1_path = f"custom_results/dit/binary_mask/{stem}_full.png"
    dit2_path = f"custom_results/dit/weighted_sum_v2/{stem}_full.png"
    dit3_path = f"custom_results/dit/weighted_sum_v3/{stem}_full.png"
    dit4_path = f"custom_results/dit/weighted_sum_v4/{stem}_full.png"
    
    md = f"### {stem} (v4 대비 개선도: {row['_diff']:.3f})\n"
    md += "| GT | Sketch | GAN (SHS) | DiTv1 | DiTv2 | DiTv3 | DiTv4 |\n"
    md += "|:--:|:------:|:---------:|:-----:|:-----:|:-----:|:-----:|\n"
    md += f"| ![]({gt_path}) | ![]({sk_path}) | ![]({gan_path}) | ![]({dit1_path}) | ![]({dit2_path}) | ![]({dit3_path}) | ![]({dit4_path}) |\n\n"
    
    md += "| | GAN | DiTv1 | DiTv2 | DiTv3 | DiTv4 |\n"
    md += "|--|:---:|:-----:|:-----:|:-----:|:-----:|\n"
    
    def fmt(v):
        try:
            fval = float(v)
            if math.isnan(fval):
                return "N/A"
            return f"{fval:.3f}"
        except:
            return "N/A"
            
    gan_p, dit1_p, dit2_p, dit3_p, dit4_p = fmt(row.get('gan_psnr')), fmt(row.get('dit1_psnr')), fmt(row.get('dit2_psnr')), fmt(row.get('dit3_psnr')), fmt(row.get('dit4_psnr'))
    gan_s, dit1_s, dit2_s, dit3_s, dit4_s = fmt(row.get('gan_ssim')), fmt(row.get('dit1_ssim')), fmt(row.get('dit2_ssim')), fmt(row.get('dit3_ssim')), fmt(row.get('dit4_ssim'))
    gan_l, dit1_l, dit2_l, dit3_l, dit4_l = fmt(row.get('gan_lpips')), fmt(row.get('dit1_lpips')), fmt(row.get('dit2_lpips')), fmt(row.get('dit3_lpips')), fmt(row.get('dit4_lpips'))
    
    # PSNR max
    psnrs = [float(x) if x != "N/A" else -1 for x in [gan_p, dit1_p, dit2_p, dit3_p, dit4_p]]
    max_p = max(psnrs)
    p_strs = [f"**{x}**" if (x != "N/A" and float(x) == max_p) else x for x in [gan_p, dit1_p, dit2_p, dit3_p, dit4_p]]
    
    # SSIM max
    ssims = [float(x) if x != "N/A" else -1 for x in [gan_s, dit1_s, dit2_s, dit3_s, dit4_s]]
    max_s = max(ssims)
    s_strs = [f"**{x}**" if (x != "N/A" and float(x) == max_s) else x for x in [gan_s, dit1_s, dit2_s, dit3_s, dit4_s]]
    
    # LPIPS min
    lpips_vals = [float(x) if x != "N/A" else 999 for x in [gan_l, dit1_l, dit2_l, dit3_l, dit4_l]]
    min_l = min(lpips_vals)
    l_strs = [f"**{x}**" if (x != "N/A" and float(x) == min_l) else x for x in [gan_l, dit1_l, dit2_l, dit3_l, dit4_l]]
    
    md += f"| PSNR ↑ | {p_strs[0]} | {p_strs[1]} | {p_strs[2]} | {p_strs[3]} | {p_strs[4]} |\n"
    md += f"| SSIM ↑ | {s_strs[0]} | {s_strs[1]} | {s_strs[2]} | {s_strs[3]} | {s_strs[4]} |\n"
    md += f"| LPIPS ↓ | {l_strs[0]} | {l_strs[1]} | {l_strs[2]} | {l_strs[3]} | {l_strs[4]} |\n\n"
    md += "---\n\n"
    return md

def main():
    csv_path = os.path.join("eval_results", "all_per_image.csv")
    out_path = "evaluation.md"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Clean up old generated content in evaluation.md
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # "## Top 20 Best Examples"가 나오는 위치를 찾아서 그 위까지만 남김
        truncate_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("## Top 20 Best Examples"):
                truncate_idx = i
                break
        
        # Strip trailing newlines just before the truncation point
        while truncate_idx > 0 and lines[truncate_idx-1].strip() == "":
            truncate_idx -= 1
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines[:truncate_idx])

    # 2. Process CSV
    rows = read_csv(csv_path)
    valid_rows = []
    for r in rows:
        try:
            dit4_l = float(r.get('dit4_lpips', ''))
            gan_l = float(r.get('gan_lpips', ''))
            dit1_l = float(r.get('dit1_lpips', ''))
            dit2_l = float(r.get('dit2_lpips', ''))
            dit3_l = float(r.get('dit3_lpips', ''))
            
            best_other_l = min(gan_l, dit1_l, dit2_l, dit3_l)
            diff = best_other_l - dit4_l
            
            r['_diff'] = diff
            valid_rows.append(r)
        except:
            pass
                
    valid_rows.sort(key=lambda x: x['_diff'], reverse=True)
    
    top_20 = valid_rows[:20]
    bottom_20 = valid_rows[-20:][::-1] 
    
    # 3. Generate Markdown
    out_md = "\n\n## Top 20 Best Examples (상대 평가: 타 모델 대비 v4 LPIPS 개선도 기준)\n\n"
    for r in top_20:
        out_md += format_row(r)
        
    out_md += "## Top 20 Worst Examples (상대 평가: 타 모델 대비 v4 LPIPS 악화도 기준)\n\n"
    for r in bottom_20:
        out_md += format_row(r)
        
    # 4. Append to file
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(out_md)
        
    print(f"Successfully cleaned old data and appended 40 new examples to {out_path}")

if __name__ == "__main__":
    main()
