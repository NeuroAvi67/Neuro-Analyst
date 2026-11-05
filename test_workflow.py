"""End-to-end tests: python -m unittest discover tests"""
import io, time, unittest, json
import numpy as np, pandas as pd
from backend.app import app

def make_csv(n=150, shift=0.0, seed=1):
    r = np.random.RandomState(seed); x = r.normal(50, 10, n)
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"), "x": x + shift, "y": 2 * x + r.normal(0, 5, n), "g": r.choice(["a", "b"], n)})
    d.loc[3, "x"] = np.nan; d.loc[5, "y"] = 5000
    return io.BytesIO(pd.concat([d, d.iloc[[7]]]).to_csv(index=False).encode())

class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = app.test_client()
        cls.ids = [cls.c.post("/api/upload", data={"files": (make_csv(seed=s, shift=sh), f"d{s}.csv")}, content_type="multipart/form-data").get_json()["files"][0]["id"] for s, sh in ((1, 0), (2, 8))]

    def test_rejects_bad_files(self):
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data").get_json()
        self.assertIn("Unsupported", r["files"][0]["error"])
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(b""), "e.csv")}, content_type="multipart/form-data").get_json()
        self.assertIn("empty", r["files"][0]["error"])

    def test_path_traversal_blocked(self):
        self.assertEqual(self.c.post("/api/clean", json={"file_id": "../../etc/passwd"}).status_code, 400)

    def test_cleaning_is_non_destructive_and_logged(self):
        r = self.c.post("/api/clean", json={"file_id": self.ids[0]}).get_json()
        self.assertEqual(r["before"]["duplicate_rows"], 1); self.assertEqual(r["after"]["duplicate_rows"], 0)
        self.assertEqual(r["after"]["missing_cells"], 0); self.assertTrue(r["log"])

    def test_stats_match_pandas(self):
        self.c.post("/api/clean", json={"file_id": self.ids[0]})
        s = self.c.post("/api/stats", json={"file_id": self.ids[0]}).get_json()
        import pickle, glob
        from backend import config
        df = pd.read_pickle(config.CLEAN_DIR / f"{self.ids[0]}.pkl")
        self.assertAlmostEqual(s["descriptive"]["x"]["mean"], df["x"].mean(), places=4)
        self.assertAlmostEqual(s["correlations"][0]["pearson_r"], df["x"].corr(df["y"]), places=4)

    def test_full_pipeline_report_exports_and_verification(self):
        jid = self.c.post("/api/analyze", json={"file_id": self.ids[0], "instruction": "scientific report", "provider": "offline"}).get_json()["job_id"]
        for _ in range(100):
            j = self.c.get(f"/api/jobs/{jid}").get_json()
            if j["status"] in ("done", "error"): break
            time.sleep(.2)
        self.assertEqual(j["status"], "done", j)
        rep = self.c.get(f"/api/reports/{jid}").get_json()
        self.assertEqual(rep["result"]["verification"]["status"], "verified")
        for fmt, magic in (("pdf", b"%PDF"), ("html", b"<!doctype"), ("md", b"# ")):
            r = self.c.get(f"/api/reports/{jid}/export?format={fmt}"); self.assertEqual(r.status_code, 200); self.assertTrue(r.data.startswith(magic))

    def test_ai_output_with_invented_numbers_is_rejected(self):
        from backend.ai import interpreter as I
        facts = [{"id": "F1", "text": "x and y have r = 0.958 (n = 150)."}]
        self.assertEqual(I.verify_narrative("r = 0.958 [F1]", facts)["status"], "verified")
        self.assertEqual(I.verify_narrative("sales grew 87.5%", facts)["status"], "unverified_numbers")

    def test_compare_and_library(self):
        c = self.c.post("/api/compare", json={"file_ids": self.ids}).get_json()["comparison"]
        self.assertTrue(c["numeric"]); self.assertTrue(c["assumptions"])
        self.assertTrue(self.c.get("/api/library?q=d1").get_json()["datasets"])

    def test_delete_dataset(self):
        up = self.c.post("/api/upload", data={"files": (make_csv(seed=9), "tmp.csv")}, content_type="multipart/form-data").get_json()["files"][0]["id"]
        self.assertTrue(self.c.delete(f"/api/datasets/{up}").get_json()["ok"])
        self.assertEqual(self.c.post("/api/analyze", json={"file_id": up}).status_code, 400)

if __name__ == "__main__":
    unittest.main()
