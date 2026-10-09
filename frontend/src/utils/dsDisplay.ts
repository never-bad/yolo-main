/**
 * 数据集显示名统一口径（UX-5/6/7）
 * 主标题 = display_name（业务名，不以 ds_ 开头）> 上传文件名去扩展名 > dataset_id
 */
export function dsTitle(ds: any): string {
  const dn = String(ds?.display_name || '').trim()
  const fn = String(ds?.filename || '').trim()
  if (dn && !dn.startsWith('ds_')) return dn
  return fn ? fn.replace(/\.[^.]+$/, '') : (dn || String(ds?.dataset_id || ''))
}
