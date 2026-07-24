# 20260724 Session — Chain 14 + Chain 15 auto-generated report

**Auto-generated at pipeline completion.**

Chain 14: 101 conditions
Chain 15: 105 conditions (batch sweep)

## Chain 14

| condition | n | cudaFree_ms | l1_mean_ms | ai_metric |
|---|---:|---:|---:|---:|
| cfgA_CP_bert_b1 | 3 | 1894 | 38.9 | ai_fwd_per_s=80.2 |
| cfgA_CP_chanpred | 3 | 1675 | 37.1 |  |
| cfgA_CP_embed_lookup | 3 | 1682 | 37.1 | ai_effective_bw_gbps=8.0 |
| cfgA_CP_hbm_stress | 3 | 1844 | 38.3 |  |
| cfgA_CP_idle | 3 | 1897 | 38.9 |  |
| cfgA_CP_memcpy_loop | 3 | 1889 | 38.9 | ai_rate_per_s=102420.7 |
| cfgA_CP_nrx | 3 | 1910 | 38.9 |  |
| cfgA_CP_qwen_chat_b1 | 3 | 2052 | 40.4 |  |
| cfgA_CP_qwen_llm_cross | 3 | 2108 | 40.8 |  |
| cfgA_CP_qwen_rag | 3 | 1840 | 38.5 |  |
| cfgA_CP_qwen_vl | 3 | 1812 | 38.2 |  |
| cfgA_CP_whisper | 3 | 1846 | 38.2 |  |
| cfgA_CP_whisper_stream_b1 | 3 | 2012 | 40.3 | ai_rtf=14.6 |
| cfgA_SP0_baseline | 3 | 1897 | 38.9 |  |
| cfgA_SP_bert_b1_MPSoff | 3 | 1685 | 37.1 |  |
| cfgA_SP_bert_b1_MPSon | 3 | 1895 | 38.9 |  |
| cfgA_SP_chanpred_MPSoff | 3 | 12004 | 161.4 |  |
| cfgA_SP_chanpred_MPSon | 3 | 1900 | 38.9 |  |
| cfgA_SP_embed_lookup_MPSoff | 3 | 4528 | 72.1 |  |
| cfgA_SP_embed_lookup_MPSon | 3 | 1764 | 37.3 |  |
| cfgA_SP_hbm_stress_MPSoff | 3 | 2122 | 40.9 |  |
| cfgA_SP_hbm_stress_MPSon | 3 | 2051 | 40.5 |  |
| cfgA_SP_memcpy_loop_MPSoff | 3 | 5327 | 81.9 |  |
| cfgA_SP_memcpy_loop_MPSon | 3 | 1963 | 39.1 |  |
| cfgA_SP_nrx_MPSoff | 3 | 11802 | 138.3 |  |
| cfgA_SP_nrx_MPSon | 3 | 1719 | 37.3 |  |
| cfgA_SP_qwen_chat_b1_MPSoff | 3 | 2058 | 41.7 |  |
| cfgA_SP_qwen_chat_b1_MPSon | 3 | 2032 | 41.4 |  |
| cfgA_SP_qwen_rag_MPSoff | 3 | 1894 | 40.0 |  |
| cfgA_SP_qwen_rag_MPSon | 3 | 1988 | 40.9 |  |
| cfgA_SP_qwen_vl_MPSoff | 3 | 1676 | 37.2 |  |
| cfgA_SP_qwen_vl_MPSon | 3 | 1848 | 38.4 |  |
| cfgA_SP_whisper_MPSoff | 3 | 1756 | 37.8 |  |
| cfgA_SP_whisper_MPSon | 3 | 2111 | 40.9 |  |
| cfgA_SP_whisper_stream_b1_MPSoff | 3 | 1787 | 38.8 |  |
| cfgA_SP_whisper_stream_b1_MPSon | 3 | 2109 | 41.0 |  |
| cfgB_SP0_baseline | 3 | 1814 | 38.0 |  |
| cfgB_SP_bert_b1_MPSoff | 3 | 1493 | 35.8 |  |
| cfgB_SP_bert_b1_MPSon | 3 | 1632 | 36.6 |  |
| cfgB_SP_chanpred_MPSoff | 3 | 7701 | 114.3 |  |
| cfgB_SP_chanpred_MPSon | 3 | 1579 | 36.0 |  |
| cfgB_SP_embed_lookup_MPSoff | 3 | 4890 | 77.1 |  |
| cfgB_SP_embed_lookup_MPSon | 3 | 1704 | 37.0 |  |
| cfgB_SP_hbm_stress_MPSoff | 3 | 5 | 0.0 |  |
| cfgB_SP_hbm_stress_MPSon | 3 | 5 | 0.0 |  |
| cfgB_SP_memcpy_loop_MPSoff | 3 | 5855 | 86.1 |  |
| cfgB_SP_memcpy_loop_MPSon | 3 | 1924 | 39.3 |  |
| cfgB_SP_nrx_MPSoff | 3 | 12118 | 124.7 |  |
| cfgB_SP_nrx_MPSon | 3 | 1771 | 38.0 |  |
| cfgB_SP_qwen_chat_b1_MPSoff | 3 | 1720 | 39.1 |  |
| cfgB_SP_qwen_chat_b1_MPSon | 3 | 1617 | 38.8 |  |
| cfgB_SP_qwen_rag_MPSoff | 3 | 1728 | 39.2 |  |
| cfgB_SP_qwen_rag_MPSon | 3 | 1530 | 38.5 |  |
| cfgB_SP_qwen_vl_MPSoff | 3 | 1435 | 34.6 |  |
| cfgB_SP_qwen_vl_MPSon | 3 | 1794 | 38.4 |  |
| cfgB_SP_whisper_MPSoff | 3 | 1856 | 39.8 |  |
| cfgB_SP_whisper_MPSon | 3 | 1390 | 34.6 |  |
| cfgB_SP_whisper_stream_b1_MPSoff | 3 | 1458 | 35.1 |  |
| cfgB_SP_whisper_stream_b1_MPSon | 3 | 1794 | 37.9 |  |
| cfgC_CP1g_chanpred | 3 | 1851 | 38.4 |  |
| cfgC_CP1g_hbm_stress | 3 | 1901 | 39.1 |  |
| cfgC_CP1g_idle | 3 | 1909 | 38.9 |  |
| cfgC_CP1g_nrx | 3 | 2015 | 39.7 |  |
| cfgC_CP1g_qwen_llm_cross | 3 | 2071 | 40.4 |  |
| cfgC_CP1g_whisper | 3 | 1903 | 38.9 |  |
| cfgC_CP_bert_b1 | 3 | 1890 | 38.8 | ai_fwd_per_s=79.6 |
| cfgC_CP_chanpred | 3 | 2053 | 40.4 |  |
| cfgC_CP_embed_lookup | 3 | 1900 | 39.1 | ai_effective_bw_gbps=7.9 |
| cfgC_CP_hbm_stress | 3 | 1696 | 36.9 |  |
| cfgC_CP_idle | 3 | 2112 | 40.9 |  |
| cfgC_CP_memcpy_loop | 3 | 1685 | 36.7 | ai_rate_per_s=102934.0 |
| cfgC_CP_nrx | 3 | 2113 | 41.1 |  |
| cfgC_CP_qwen_chat_b1 | 3 | 1706 | 36.8 |  |
| cfgC_CP_qwen_llm_cross | 3 | 1689 | 36.9 |  |
| cfgC_CP_qwen_rag | 3 | 1702 | 36.9 |  |
| cfgC_CP_qwen_vl | 3 | 1982 | 39.8 |  |
| cfgC_CP_whisper | 3 | 1841 | 38.5 |  |
| cfgC_CP_whisper_stream_b1 | 3 | 1837 | 38.4 | ai_rtf=11.8 |
| cfgC_SP0_baseline | 3 | 1700 | 36.7 |  |
| cfgC_SP_bert_b1_MPSoff | 3 | 1751 | 37.9 |  |
| cfgC_SP_bert_b1_MPSon | 3 | 2125 | 41.2 |  |
| cfgC_SP_chanpred_MPSoff | 3 | 12780 | 168.8 |  |
| cfgC_SP_chanpred_MPSon | 3 | 1907 | 39.1 |  |
| cfgC_SP_embed_lookup_MPSoff | 3 | 4592 | 73.2 |  |
| cfgC_SP_embed_lookup_MPSon | 3 | 1987 | 39.7 |  |
| cfgC_SP_hbm_stress_MPSoff | 3 | 1851 | 38.4 |  |
| cfgC_SP_hbm_stress_MPSon | 3 | 1693 | 36.9 |  |
| cfgC_SP_memcpy_loop_MPSoff | 3 | 5341 | 82.1 |  |
| cfgC_SP_memcpy_loop_MPSon | 3 | 1781 | 37.6 |  |
| cfgC_SP_nrx_MPSoff | 3 | 11328 | 174.6 |  |
| cfgC_SP_nrx_MPSon | 3 | 2093 | 40.8 |  |
| cfgC_SP_qwen_chat_b1_MPSoff | 3 | 2097 | 42.3 |  |
| cfgC_SP_qwen_chat_b1_MPSon | 3 | 2432 | 45.5 |  |
| cfgC_SP_qwen_rag_MPSoff | 3 | 1929 | 40.3 |  |
| cfgC_SP_qwen_rag_MPSon | 3 | 2077 | 41.4 |  |
| cfgC_SP_qwen_vl_MPSoff | 3 | 2207 | 42.1 |  |
| cfgC_SP_qwen_vl_MPSon | 3 | 2124 | 40.9 |  |
| cfgC_SP_whisper_MPSoff | 3 | 2040 | 40.7 |  |
| cfgC_SP_whisper_MPSon | 3 | 1910 | 38.9 |  |
| cfgC_SP_whisper_stream_b1_MPSoff | 3 | 1975 | 40.3 |  |
| cfgC_SP_whisper_stream_b1_MPSon | 3 | 1694 | 37.0 |  |

## Chain 15

| condition | n | cudaFree_ms | l1_mean_ms | ai_metric |
|---|---:|---:|---:|---:|
| cfgA_SP0_baseline | 3 | 2006 | 39.8 |  |
| cfgA_SP_bert_b16_MPSoff | 3 | 2688 | 50.5 |  |
| cfgA_SP_bert_b16_MPSon | 3 | 1899 | 39.1 |  |
| cfgA_SP_bert_b1_MPSoff | 3 | 1886 | 39.6 |  |
| cfgA_SP_bert_b1_MPSon | 3 | 1902 | 38.8 |  |
| cfgA_SP_bert_b4_MPSoff | 3 | 2297 | 45.2 |  |
| cfgA_SP_bert_b4_MPSon | 3 | 2114 | 40.9 |  |
| cfgA_SP_bert_b64_MPSoff | 3 | 1739 | 37.8 |  |
| cfgA_SP_bert_b64_MPSon | 3 | 1904 | 38.9 |  |
| cfgA_SP_qwen_chat_b16_MPSoff | 3 | 1905 | 39.7 |  |
| cfgA_SP_qwen_chat_b16_MPSon | 3 | 2167 | 42.9 |  |
| cfgA_SP_qwen_chat_b1_MPSoff | 3 | 1889 | 39.9 |  |
| cfgA_SP_qwen_chat_b1_MPSon | 3 | 2023 | 41.5 |  |
| cfgA_SP_qwen_chat_b2_MPSoff | 3 | 1890 | 39.8 |  |
| cfgA_SP_qwen_chat_b2_MPSon | 3 | 2026 | 41.1 |  |
| cfgA_SP_qwen_chat_b32_MPSoff | 3 | 2009 | 41.2 |  |
| cfgA_SP_qwen_chat_b32_MPSon | 3 | 2013 | 40.8 |  |
| cfgA_SP_qwen_chat_b4_MPSoff | 3 | 1884 | 39.8 |  |
| cfgA_SP_qwen_chat_b4_MPSon | 3 | 2358 | 44.5 |  |
| cfgA_SP_qwen_chat_b8_MPSoff | 3 | 1888 | 40.0 |  |
| cfgA_SP_qwen_chat_b8_MPSon | 3 | 2011 | 41.3 |  |
| cfgA_SP_vl_b1_MPSoff | 3 | 1689 | 36.9 |  |
| cfgA_SP_vl_b1_MPSon | 3 | 2287 | 42.3 |  |
| cfgA_SP_vl_b2_MPSoff | 3 | 2011 | 39.9 |  |
| cfgA_SP_vl_b2_MPSon | 3 | 1685 | 37.0 |  |
| cfgA_SP_vl_b4_MPSoff | 3 | 1886 | 38.9 |  |
| cfgA_SP_vl_b4_MPSon | 3 | 1904 | 38.9 |  |
| cfgA_SP_whisper_b1_MPSoff | 3 | 1769 | 38.3 |  |
| cfgA_SP_whisper_b1_MPSon | 3 | 1685 | 37.1 |  |
| cfgA_SP_whisper_b2_MPSoff | 3 | 1851 | 39.3 |  |
| cfgA_SP_whisper_b2_MPSon | 3 | 1914 | 38.9 |  |
| cfgA_SP_whisper_b4_MPSoff | 3 | 2161 | 42.1 |  |
| cfgA_SP_whisper_b4_MPSon | 3 | 1914 | 38.7 |  |
| cfgA_SP_whisper_b8_MPSoff | 3 | 2034 | 41.3 |  |
| cfgA_SP_whisper_b8_MPSon | 3 | 2119 | 40.7 |  |
| cfgB_SP0_baseline | 3 | 1754 | 37.5 |  |
| cfgB_SP_bert_b16_MPSoff | 3 | 2402 | 45.9 |  |
| cfgB_SP_bert_b16_MPSon | 3 | 1636 | 36.6 |  |
| cfgB_SP_bert_b1_MPSoff | 3 | 1842 | 38.8 |  |
| cfgB_SP_bert_b1_MPSon | 3 | 1406 | 34.5 |  |
| cfgB_SP_bert_b4_MPSoff | 3 | 1454 | 35.5 |  |
| cfgB_SP_bert_b4_MPSon | 3 | 1845 | 38.7 |  |
| cfgB_SP_bert_b64_MPSoff | 3 | 1693 | 37.4 |  |
| cfgB_SP_bert_b64_MPSon | 3 | 1636 | 36.4 |  |
| cfgB_SP_qwen_chat_b16_MPSoff | 3 | 1646 | 38.9 |  |
| cfgB_SP_qwen_chat_b16_MPSon | 3 | 1544 | 38.6 |  |
| cfgB_SP_qwen_chat_b1_MPSoff | 3 | 1544 | 37.8 |  |
| cfgB_SP_qwen_chat_b1_MPSon | 3 | 1516 | 37.8 |  |
| cfgB_SP_qwen_chat_b2_MPSoff | 3 | 1638 | 38.7 |  |
| cfgB_SP_qwen_chat_b2_MPSon | 3 | 1485 | 37.4 |  |
| cfgB_SP_qwen_chat_b32_MPSoff | 3 | 1612 | 38.6 |  |
| cfgB_SP_qwen_chat_b32_MPSon | 3 | 1738 | 39.1 |  |
| cfgB_SP_qwen_chat_b4_MPSoff | 3 | 1530 | 37.6 |  |
| cfgB_SP_qwen_chat_b4_MPSon | 3 | 1467 | 37.4 |  |
| cfgB_SP_qwen_chat_b8_MPSoff | 3 | 1533 | 38.1 |  |
| cfgB_SP_qwen_chat_b8_MPSon | 3 | 1487 | 37.9 |  |
| cfgB_SP_vl_b1_MPSoff | 3 | 1659 | 36.5 |  |
| cfgB_SP_vl_b1_MPSon | 3 | 1426 | 34.5 |  |
| cfgB_SP_vl_b2_MPSoff | 3 | 1862 | 38.5 |  |
| cfgB_SP_vl_b2_MPSon | 3 | 1796 | 38.2 |  |
| cfgB_SP_vl_b4_MPSoff | 3 | 1627 | 36.6 |  |
| cfgB_SP_vl_b4_MPSon | 3 | 1805 | 38.2 |  |
| cfgB_SP_whisper_b1_MPSoff | 3 | 1655 | 37.2 |  |
| cfgB_SP_whisper_b1_MPSon | 3 | 1424 | 34.6 |  |
| cfgB_SP_whisper_b2_MPSoff | 3 | 1641 | 36.6 |  |
| cfgB_SP_whisper_b2_MPSon | 3 | 1855 | 38.7 |  |
| cfgB_SP_whisper_b4_MPSoff | 3 | 1416 | 34.5 |  |
| cfgB_SP_whisper_b4_MPSon | 3 | 1974 | 39.7 |  |
| cfgB_SP_whisper_b8_MPSoff | 3 | 1496 | 35.6 |  |
| cfgB_SP_whisper_b8_MPSon | 3 | 1842 | 38.4 |  |
| cfgC_SP0_baseline | 3 | 1846 | 38.3 |  |
| cfgC_SP_bert_b16_MPSoff | 3 | 1944 | 39.8 |  |
| cfgC_SP_bert_b16_MPSon | 3 | 1917 | 38.9 |  |
| cfgC_SP_bert_b1_MPSoff | 3 | 2076 | 41.6 |  |
| cfgC_SP_bert_b1_MPSon | 3 | 2121 | 40.9 |  |
| cfgC_SP_bert_b4_MPSoff | 3 | 2983 | 55.0 |  |
| cfgC_SP_bert_b4_MPSon | 3 | 1912 | 39.0 |  |
| cfgC_SP_bert_b64_MPSoff | 3 | 2429 | 46.3 |  |
| cfgC_SP_bert_b64_MPSon | 3 | 1929 | 38.8 |  |
| cfgC_SP_qwen_chat_b16_MPSoff | 3 | 2104 | 42.1 |  |
| cfgC_SP_qwen_chat_b16_MPSon | 3 | 1886 | 39.7 |  |
| cfgC_SP_qwen_chat_b1_MPSoff | 3 | 2114 | 41.9 |  |
| cfgC_SP_qwen_chat_b1_MPSon | 3 | 1906 | 39.9 |  |
| cfgC_SP_qwen_chat_b2_MPSoff | 3 | 2241 | 43.7 |  |
| cfgC_SP_qwen_chat_b2_MPSon | 3 | 2067 | 41.5 |  |
| cfgC_SP_qwen_chat_b32_MPSoff | 3 | 2119 | 41.9 |  |
| cfgC_SP_qwen_chat_b32_MPSon | 3 | 2036 | 41.6 |  |
| cfgC_SP_qwen_chat_b4_MPSoff | 3 | 2124 | 42.3 |  |
| cfgC_SP_qwen_chat_b4_MPSon | 3 | 2185 | 43.6 |  |
| cfgC_SP_qwen_chat_b8_MPSoff | 3 | 2043 | 41.6 |  |
| cfgC_SP_qwen_chat_b8_MPSon | 3 | 2048 | 41.4 |  |
| cfgC_SP_vl_b1_MPSoff | 3 | 1689 | 37.1 |  |
| cfgC_SP_vl_b1_MPSon | 3 | 2281 | 42.5 |  |
| cfgC_SP_vl_b2_MPSoff | 3 | 1683 | 36.9 |  |
| cfgC_SP_vl_b2_MPSon | 3 | 1920 | 38.9 |  |
| cfgC_SP_vl_b4_MPSoff | 3 | 1687 | 37.0 |  |
| cfgC_SP_vl_b4_MPSon | 3 | 2118 | 41.1 |  |
| cfgC_SP_whisper_b1_MPSoff | 3 | 1987 | 40.1 |  |
| cfgC_SP_whisper_b1_MPSon | 3 | 1714 | 36.8 |  |
| cfgC_SP_whisper_b2_MPSoff | 3 | 1982 | 40.3 |  |
| cfgC_SP_whisper_b2_MPSon | 3 | 1858 | 38.5 |  |
| cfgC_SP_whisper_b4_MPSoff | 3 | 1753 | 37.7 |  |
| cfgC_SP_whisper_b4_MPSon | 3 | 1919 | 38.9 |  |
| cfgC_SP_whisper_b8_MPSoff | 3 | 1693 | 36.8 |  |
| cfgC_SP_whisper_b8_MPSon | 3 | 2244 | 41.9 |  |

