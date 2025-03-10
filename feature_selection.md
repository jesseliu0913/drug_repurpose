# 药物再利用特征选择与理由

## 药物特征 (Drug Features)

### 1. **`drug_indication`（适应症，原`indication`）**
   - **理由**：适应症明确指出了药物已被批准用于治疗的疾病或症状。这是药物再利用的起点，因为已知的治疗用途可以提示药物在相似疾病中的潜在应用。例如，一种治疗炎症性肠病的药物可能对其他炎症性疾病（如类风湿关节炎）有效。
   - **作用**：提供药物的已验证治疗领域，作为寻找新用途的参考基础。

### 2. **`drug_mechanism`（作用机制，原`mechanism_of_action`）**
   - **理由**：作用机制描述了药物在分子水平上的作用方式，例如抑制某个酶或激活某个受体。在药物再利用中，作用机制至关重要，因为相同的机制可能适用于不同的疾病。例如，抑制血管生成的药物原本用于癌症治疗，但也可用于湿性年龄相关性黄斑变性（AMD）。
   - **作用**：将药物的生物学作用与新疾病的分子病理机制相匹配。

### 3. **`drug_pharmacodynamics`（药效学，原`pharmacodynamics`）**
   - **理由**：药效学揭示了药物对身体产生的生物学效应，例如降低血压或抑制炎症。这些效应可以直接与疾病的临床表现或生物标志物联系起来，帮助判断药物是否可能缓解新疾病的症状。
   - **作用**：提供药物影响生理过程的信息，支持基于表型的再利用研究。

### 4. **`pathway`（通路）** ❌
   - **理由**：生物通路（如Wnt信号通路）是药物作用的分子背景。通过了解药物影响的通路，可以找到与特定疾病相关的通路，从而推测药物的潜在疗效。
   - **作用**：连接药物与疾病的生物学机制，支持基于网络药理学的再利用分析。

### 5. **ATC分类系统** ✅ 
   - **理由**：ATC（解剖学治疗化学）分类系统根据药物的解剖学、治疗和化学特性对其进行分级：
     - **`drug_anatomical_group`**（原`Anatomical Group`/`atc_1`）：第一层级，表示药物作用于人体的哪个解剖系统（如心血管、消化系统）。
     - **`drug_therapeutic_group`**（原`Therapeutic Group`/`atc_2`）：第二层级，表示药物的治疗用途（如抗高血压、抗抑郁等）。
     - **`drug_pharmacological_group`**（原`Pharmacological Group`/`atc_3`）：第三层级，表示药物的具体药理作用（如β受体阻滞剂、ACE抑制剂）。
     - **`drug_chemical_and_functional_group`**（原`Chemical and Functional Group`/`atc_4`）：第四层级，表示药物的具体化学类型（如卡托普利、赖诺普利等）。
     这些层级提供了药物的系统性分类，ATC级别相似的药物可能具有相似的治疗特性或作用机制，可用于推测其在新疾病中的用途。
   - **作用**：帮助识别具有相似特性的药物，支持基于分类的再利用筛选。

### 6. **`drug_description`（描述，原`description`）**
   - **理由**：药物的文本描述包含丰富的背景信息，例如开发历史、已知副作用和临床试验结果。通过自然语言处理（NLP）技术，可以从中提取模式或线索，为再利用提供补充信息。
   - **作用**：提供多模态数据，增强结构化特征的分析能力。

### 7. **`drug_category`（类别，原`category`）**✅
   - **理由**：类别（如抗生素、抗癌药）为药物提供了额外的分类维度，与ATC分类互补。这种分组有助于识别相似药物并进行比较。
   - **作用**：支持基于类别的筛选和候选药物识别。

### 8. **药物类型（原`group`）** ✅ 一组one-hot编码特征：
   - **理由**：药物类型信息反映了药物的开发阶段。已获批药物在再利用中更具优势，因为其安全性数据更完善，临床应用风险较低。
   - **特征集**：
     - `drug_is_approved`（已批准）
     - `drug_is_investigational`（在研）
     - `drug_is_experimental`（实验性）
     - `drug_is_vet_approved`（兽用）
     - `drug_is_nutraceutical`（营养补充剂）
     - `drug_is_illicit`（非法药物）
     - `drug_is_withdrawn`（已撤市）
   - **作用**：优先选择已获批药物，降低再利用的开发成本和风险。

---

## 疾病特征（Disease Features）

### 1. **`disease_mondo_definition`（MONDO定义，原`mondo_definition`）**
   - **理由**：MONDO（Monarch Disease Ontology）定义提供了疾病的简洁、标准化描述，便于快速了解疾病的基本特征。
   - **作用**：作为疾病的统一标识，支持跨数据库的整合和比较。

### 2. **`disease_umls_description`（UMLS描述，原`umls_description`）**
   - **理由**：UMLS（统一医学语言系统）描述包含疾病的详细文本信息，如病因、症状和诊断等。通过NLP技术挖掘这些信息，可以分析疾病相似性并匹配潜在药物。
   - **作用**：提供多模态信息，揭示疾病的潜在生物学联系。

### 3. **`disease_clinical_description`（Orphanet临床描述，原`orphanet_clinical_description`）**
   - **理由**：Orphanet专注于罕见病，其临床描述提供了疾病的详细表型和特征。这些信息对于理解疾病的临床表现和治疗靶点尤为重要。
   - **作用**：支持基于表型的药物再利用，尤其适用于罕见病研究。

### 4. **`disease_symptoms`（Mayo Clinic症状，原`mayo_symptoms`）**
   - **理由**：Mayo Clinic列出的症状是疾病表型的具体体现。症状可以与药物的药效学效应或副作用匹配，从而判断药物是否能缓解特定症状🌟。
   - **作用**：帮助基于表型寻找合适的候选药物。

### 5. **`disease_causes`（Mayo Clinic病因，原`mayo_causes`）**
   - **理由**：病因信息揭示了疾病的起源和发生机制。理解病因有助于识别潜在治疗靶点，并将药物作用机制与疾病联系起来。
   - **作用**：支持基于机制的药物再利用。

### 6. **`disease_risk_factors`（Mayo Clinic风险因素，原`mayo_risk_factors`）**
   - **理由**：风险因素指出了疾病发生或恶化的相关条件。这些因素可能与药物的作用机制或副作用相关，从而指导药物的再利用方向。
   - **作用**：帮助识别药物在预防或治疗风险因素相关疾病中的潜力。

---

## 关系特征
### 1. **`relation_subtype`（关系子类型，原`effect_type`）**
   - **理由**：描述药物和疾病/表型之间的具体关系类型，如适应症、禁忌症或离标用药。
   - **作用**：为模型提供药物-疾病关系的性质，有助于区分不同类型的治疗关系。

---

## 为什么不选择其他特征？

为了保持选择的针对性，我排除了一些非核心特征：

### 药物特征中的非核心特征
- **`half_life`（半衰期）**、**`protein_binding`（蛋白结合率）**、**`molecular_weight`（分子量）**、**`tpsa`（极性表面积）**、**`clogp`（脂溶性）**：这些药代动力学特征影响药物的吸收和代谢，在药物开发后期优化候选药物时更重要，但在再利用的早期筛选中不是关键。
- **`state`（状态）**：药物的物理状态（如固体、液体）对再利用的直接影响有限，除非涉及特定给药方式。

### 疾病特征中的非核心特征
- **`orphanet_prevalence`（流行率）**、**`orphanet_epidemiology`（流行病学）**：这些信息对罕见病研究或公共卫生决策有用，但在药物与疾病机制匹配中作用不大。
- **`mayo_complications`（并发症）**、**`mayo_prevention`（预防）**、**`mayo_see_doc`（就医建议）**：这些特征对临床管理有意义，但在建立药物与疾病的生物学关联时并非必需。

---

## 总结

- **药物特征**：选择`drug_indication`、`drug_mechanism`、`drug_pharmacodynamics`、`pathway`、`drug_anatomical_group`至`drug_chemical_and_functional_group`、`drug_description`、`drug_category`和药物类型one-hot特征（如`drug_is_approved`），因为它们涵盖了药物的治疗用途、作用机制和开发背景，是药物再利用的核心依据。

- **疾病特征**：选择`disease_mondo_definition`、`disease_umls_description`、`disease_clinical_description`、`disease_symptoms`、`disease_causes`和`disease_risk_factors`，因为它们提供了疾病的生物学、表型和临床信息，有助于匹配药物的作用。

这些特征结合机制、表型和文本信息，为药物再利用研究提供了全面支持。