import { useState } from 'react';
import { ScrollView, View, Text, Pressable, ActivityIndicator, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import Field from './components/Field';
import { API_URL } from './api';

const FIELDS = [
  { key: 'pregnancies', label: 'Số lần mang thai', def: '6' },
  { key: 'glucose', label: 'Glucose (mg/dL)', def: '148' },
  { key: 'blood_pressure', label: 'Huyết áp (mm Hg)', def: '72' },
  { key: 'skin_thickness', label: 'Độ dày da (mm)', def: '35' },
  { key: 'insulin', label: 'Insulin (mu U/ml)', def: '0' },
  { key: 'bmi', label: 'BMI', def: '33.6' },
  { key: 'diabetes_pedigree', label: 'Diabetes Pedigree', def: '0.627' },
  { key: 'age', label: 'Tuổi', def: '50' },
];

export default function App() {
  const [form, setForm] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.def]))
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const setValue = (key) => (text) => setForm((p) => ({ ...p, [key]: text }));

  async function predict() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const body = Object.fromEntries(FIELDS.map((f) => [f.key, parseFloat(form[f.key]) || 0]));
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError('Không kết nối được API. Kiểm tra link server trong api.js hoặc thử lại (lần đầu server có thể cần ~1 phút để khởi động).');
    }
    setLoading(false);
  }

  const positive = result?.prediction === 1;

  return (
    <>
      <StatusBar style="light" />
      <ScrollView style={styles.container} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🩺 Chẩn đoán Tiểu đường</Text>
          <Text style={styles.headerSub}>Logistic Regression • Pima Indians Diabetes</Text>
        </View>

        <View style={styles.card}>
          <View style={styles.grid}>
            {FIELDS.map((f) => (
              <Field key={f.key} label={f.label} value={form[f.key]} onChange={setValue(f.key)} />
            ))}
          </View>

          <Pressable style={({ pressed }) => [styles.button, pressed && { transform: [{ scale: 0.98 }] }]} onPress={predict} disabled={loading}>
            {loading ? (
              <View style={styles.rowCenter}>
                <ActivityIndicator color="#fff" />
                <Text style={styles.buttonText}>  Đang phân tích...</Text>
              </View>
            ) : (
              <Text style={styles.buttonText}>Xét nghiệm ngay</Text>
            )}
          </Pressable>

          {error !== '' && (
            <View style={[styles.result, { backgroundColor: '#fef9c3', borderColor: '#ca8a04' }]}>
              <Text style={[styles.resultTitle, { color: '#a16207' }]}>Lỗi kết nối</Text>
              <Text style={styles.resultMsg}>{error}</Text>
            </View>
          )}

          {result && (
            <View
              style={[
                styles.result,
                positive ? { backgroundColor: '#fee2e2', borderColor: '#dc2626' } : { backgroundColor: '#dcfce7', borderColor: '#16a34a' },
              ]}
            >
              <Text style={[styles.resultTitle, { color: positive ? '#b91c1c' : '#15803d' }]}>
                {positive ? '⚠️ Dương tính' : '✅ Âm tính'}
              </Text>
              <Text style={styles.resultMsg}>{result.message}</Text>
              <Text style={styles.confidenceLabel}>Độ tin cậy: {result.confidence_score}%</Text>
              <View style={styles.barBg}>
                <View style={[styles.barFill, { width: `${result.confidence_score}%` }]} />
              </View>
            </View>
          )}
        </View>
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#eef2ff' },
  content: { padding: 16, paddingBottom: 40 },
  header: { borderRadius: 16, padding: 20, marginBottom: 16, backgroundColor: '#2563eb' },
  headerTitle: { color: '#fff', fontSize: 21, fontWeight: '800' },
  headerSub: { color: 'rgba(255,255,255,0.85)', marginTop: 6, fontSize: 12 },
  card: { backgroundColor: '#fff', borderRadius: 16, padding: 18 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  button: { marginTop: 8, backgroundColor: '#2563eb', borderRadius: 12, paddingVertical: 15, alignItems: 'center' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  rowCenter: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  result: { marginTop: 16, padding: 18, borderRadius: 14, borderWidth: 1.5 },
  resultTitle: { fontSize: 19, fontWeight: '800', textAlign: 'center' },
  resultMsg: { color: '#475569', fontSize: 13, textAlign: 'center', marginTop: 6, lineHeight: 19 },
  confidenceLabel: { textAlign: 'center', marginTop: 10, fontSize: 13, fontWeight: '600', color: '#334155' },
  barBg: { height: 10, backgroundColor: '#e2e8f0', borderRadius: 6, overflow: 'hidden', marginTop: 8 },
  barFill: { height: '100%', backgroundColor: '#22c55e', borderRadius: 6 },
});
