"""
Exploratory Data Analysis for TB Detection System
Provides comprehensive EDA including distributions, clustering, and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class TBEDA:
    """
    Comprehensive Exploratory Data Analysis for TB Detection Data.
    Includes distributions, correlations, clustering, and advanced analytics.
    """

    def __init__(self, data_path=None, data=None):
        """
        Initialize EDA with data.

        Parameters:
        - data_path: Path to CSV file containing TB data
        - data: pandas DataFrame (alternative to data_path)
        """
        if data is not None:
            self.data = data.copy()
        elif data_path:
            self.data = pd.read_csv(data_path)
        else:
            # Generate sample data if none provided
            from utils import create_sample_data
            self.data = create_sample_data(n=10000, tb_prevalence=0.015)

        self.numeric_cols = []
        self.categorical_cols = []
        self.symptom_cols = []
        self.demographic_cols = []
        self._prepare_data()

        print(f"Loaded dataset: {len(self.data)} samples, {len(self.data.columns)} columns")
        print(f"TB prevalence: {(self.data['TB'] == '1').mean():.1%}")

    def _prepare_data(self):
        """Prepare data for analysis."""
        # Convert TB and symptoms to numeric for analysis
        symptom_columns = ['cough', 'cough_gt_2w', 'blood_in_sputum', 'fever', 'low_grade_fever',
                          'weight_loss', 'night_sweats', 'chest_pain', 'breathing_problem',
                          'fatigue', 'loss_of_appetite', 'contact_with_TB']

        self.symptom_cols = [col for col in symptom_columns if col in self.data.columns]

        # Convert symptom cols and TB to numeric
        for col in self.symptom_cols + ['TB']:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(int)

        # Demographic columns
        potential_demo_cols = ['age', 'gender', 'bmi_value', 'bmi_category', 'height', 'weight']
        self.demographic_cols = [col for col in potential_demo_cols if col in self.data.columns]

        # Convert gender to numeric for clustering
        if 'gender' in self.data.columns:
            self.data['gender_numeric'] = (self.data['gender'] == 'M').astype(int)

        # Prepare BMI categories for analysis
        if 'bmi_category' in self.data.columns:
            self.data['bmi_cat_num'] = self.data['bmi_category'].map({
                'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3
            }).fillna(1)

    def basic_statistics(self):
        """Generate comprehensive basic statistics."""
        print("\n" + "="*60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*60)

        # Dataset overview
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # TB prevalence statistics
        tb_stats = self.data['TB'].value_counts()
        print(f"\nTB Prevalence:")
        print(f"  Total Cases: {tb_stats.sum()}")
        print(f"  Positive Cases: {tb_stats[1]} ({tb_stats[1]/tb_stats.sum():.1%})")
        print(f"  Negative Cases: {tb_stats[0]} ({tb_stats[0]/tb_stats.sum():.1%})")

        # Demographic statistics
        if self.demographic_cols:
            print(f"\nDemographic Statistics:")
            for col in ['age', 'bmi_value', 'height', 'weight']:
                if col in self.data.columns:
                    print(f"  {col.title()}:")
                    print(f"    Mean: {self.data[col].mean():.1f}")
                    print(f"    Median: {self.data[col].median():.1f}")
                    print(f"    Std: {self.data[col].std():.1f}")
                    print(f"    Range: {self.data[col].min():.1f} - {self.data[col].max():.1f}")

            if 'gender' in self.data.columns:
                gender_dist = self.data['gender'].value_counts()
                print(f"  Gender: M={gender_dist['M']} ({gender_dist['M']/len(self.data):.1%}), "
                     f"F={gender_dist['F']} ({gender_dist['F']/len(self.data):.1%})")

        # Symptom prevalence
        print(f"\nSymptom Prevalence Statistics:")
        tb_positive = self.data[self.data['TB'] == 1]
        tb_negative = self.data[self.data['TB'] == 0]

        for symptom in self.symptom_cols:
            tb_rate = tb_positive[symptom].mean()
            non_tb_rate = tb_negative[symptom].mean()
            odds_ratio = tb_rate / non_tb_rate if non_tb_rate > 0 else float('inf')
            print(f"  {symptom}: TB={tb_rate:.3f}, non-TB={non_tb_rate:.3f}, OR={odds_ratio:.2f}")

    def plot_distributions(self, save_path=None):
        """Create comprehensive distribution plots."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Distribution Analysis of TB Detection Data', fontsize=16, fontweight='bold')

        # Age distribution
        if 'age' in self.data.columns:
            sns.histplot(data=self.data, x='age', hue='TB', ax=axes[0,0], alpha=0.7)
            axes[0,0].set_title('Age Distribution by TB Status')

        # BMI distribution
        if 'bmi_value' in self.data.columns:
            sns.histplot(data=self.data, x='bmi_value', hue='TB', ax=axes[0,1], alpha=0.7)
            axes[0,1].set_title('BMI Distribution by TB Status')

        # Symptom prevalence comparison
        top_symptoms = ['cough', 'cough_gt_2w', 'weight_loss', 'fatigue', 'fever']
        available_symptoms = [s for s in top_symptoms if s in self.data.columns][:4]

        if len(available_symptoms) >= 4:
            symptom_prevalence = self.data.groupby('TB')[available_symptoms].mean().T
            symptom_prevalence.plot(kind='bar', ax=axes[0,2], width=0.8)
            axes[0,2].set_title('Symptom Prevalence Comparison')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].legend(['TB Negative', 'TB Positive'])

        #BMI category distribution
        if 'bmi_category' in self.data.columns:
            tb_by_bmi = self.data.groupby(['bmi_category', 'TB']).size().unstack()
            tb_by_bmi_pct = tb_by_bmi.div(tb_by_bmi.sum(axis=1), axis=0)
            tb_by_bmi_pct.plot(kind='bar', ax=axes[0,3], stacked=True)
            axes[0,3].set_title('TB Prevalence by BMI Category')
            axes[0,3].tick_params(axis='x', rotation=45)

        # Gender distribution
        if 'gender' in self.data.columns:
            gender_tb = pd.crosstab(self.data['gender'], self.data['TB'], normalize='index')
            gender_tb.plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('TB Rate by Gender')
            axes[1,0].tick_params(axis='x', rotation=0)

        # Symptom correlation heatmap
        available_symptoms_6 = [s for s in self.symptom_cols if s in self.data.columns][:6]
        if len(available_symptoms_6) >= 4:
            symptom_corr = self.data[available_symptoms_6].corr()
            sns.heatmap(symptom_corr, annot=True, cmap='RdYlBu_r', ax=axes[1,1], center=0, fmt='.2f')
            axes[1,1].set_title('Symptom Correlation Matrix')

        # Feature importance (effect sizes)
        if self.symptom_cols:
            effect_sizes = []
            for symptom in self.symptom_cols[:6]:  # Show first 6 symptoms
                if symptom in self.data.columns:
                    tb_pos = self.data[self.data['TB'] == 1][symptom]
                    tb_neg = self.data[self.data['TB'] == 0][symptom]
                    cohens_d = (tb_pos.mean() - tb_neg.mean()) / np.sqrt((tb_pos.var() + tb_neg.var()) / 2)
                    effect_sizes.append((symptom.replace('_', ' ').title(), cohens_d))

            if effect_sizes:
                effect_df = pd.DataFrame(effect_sizes, columns=['Symptom', 'Effect Size'])
                sns.barplot(data=effect_df, x='Symptom', y='Effect Size', ax=axes[1,2])
                axes[1,2].set_title('Symptom Effect Sizes (Cohen\'s d)')
                axes[1,2].tick_params(axis='x', rotation=45)

        # Age vs BMI scatter
        if 'age' in self.data.columns and 'bmi_value' in self.data.columns:
            sns.scatterplot(data=self.data, x='age', y='bmi_value', hue='TB', ax=axes[1,3], alpha=0.6)
            axes[1,3].set_title('Age vs BMI Scatter Plot')
            axes[1,3].axhline(25, ls='--', color='orange', alpha=0.5)
            axes[1,3].axhline(30, ls='--', color='red', alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"EDA plots saved to {save_path}")
        plt.show()
        plt.close()

    def clustering_analysis(self, n_clusters_range=range(2, 8), save_path=None):
        """Perform comprehensive clustering analysis."""
        print("\n" + "="*50)
        print("CLUSTERING ANALYSIS")
        print("="*50)

        # Prepare data for clustering
        clustering_features = self.symptom_cols.copy()

        # Add demographic features if available
        if 'age' in self.data.columns:
            clustering_features.append('age')
        if 'gender_numeric' in self.data.columns:
            clustering_features.append('gender_numeric')
        if 'bmi_cat_num' in self.data.columns:
            clustering_features.append('bmi_cat_num')

        # Filter to available features
        clustering_features = [f for f in clustering_features if f in self.data.columns]

        if not clustering_features:
            print("No suitable features for clustering found.")
            return

        X = self.data[clustering_features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"Clustering on {len(clustering_features)} features: {clustering_features}")

        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        print(f"PCA explained {pca.explained_variance_ratio_[:2].sum():.1%} of variance")

        # Clustering visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('TB Data Clustering Analysis', fontsize=16, fontweight='bold')

        # K-means clusters
        scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6)
        axes[0,0].set_title('K-means Clustering (4 clusters)')
        axes[0,0].set_xlabel('PC1')
        axes[0,0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[0,0])

        # DBSCAN clustering
        dbscan = DBSCAN(eps=2.5, min_samples=50)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab10', alpha=0.6)
        axes[0,1].set_title(f'DBSCAN Clustering\n({n_clusters} clusters, {n_noise} noise)')
        axes[0,1].set_xlabel('PC1')
        axes[0,1].set_ylabel('PC2')

        # Cluster profiles heatmap
        cluster_profiles = self.data.copy()
        cluster_profiles['cluster'] = kmeans_labels
        cluster_means = cluster_profiles.groupby('cluster')[clustering_features[:8]].mean()
        sns.heatmap(cluster_means.T, annot=True, cmap='RdYlBu_r', ax=axes[0,2], center=cluster_means.values.mean())

        # TB prevalence by cluster
        tb_by_cluster = cluster_profiles.groupby('cluster')['TB'].mean()
        tb_by_cluster.plot(kind='bar', ax=axes[1,0], color='coral', alpha=0.7)
        axes[1,0].set_title('TB Prevalence by Cluster')
        axes[1,0].set_ylabel('TB Rate')
        axes[1,0].set_xlabel('Cluster')

        # Elbow plot for K-means
        distortions = []
        for k in range(2, 8):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            distortions.append(kmeans_temp.inertia_)

        axes[1,1].plot(range(2, 8), distortions, 'bo-')
        axes[1,1].set_xlabel('Number of Clusters (k)')
        axes[1,1].set_ylabel('Within-Cluster Sum of Squares')
        axes[1,1].set_title('Elbow Plot for Optimal k')
        axes[1,1].grid(True, alpha=0.3)

        # Feature importance in PCA
        feature_importance = pd.DataFrame({
            'feature': clustering_features,
            'importance': np.abs(pca.components_[0])
        }).sort_values('importance', ascending=True).tail(8)

        axes[1,2].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1,2].set_xlabel('Absolute PC1 Loading')
        axes[1,2].set_title('Top Features in PC1')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clustering analysis saved to {save_path}")
        plt.show()
        plt.close()

        print(f"\nClustering Results:")
        print(f"- K-means: 4 clusters identified")
        print(f"- DBSCAN: {n_clusters} density-based clusters found")
        print(f"- Most important features: {', '.join(feature_importance['feature'].tail(3).tolist())}")

    def generate_report(self):
        """Generate comprehensive EDA report."""
        print("🚀 Starting Comprehensive TB EDA Analysis...")
        self.basic_statistics()
        self.plot_distributions(save_path='tb_eda_distributions.png')
        self.clustering_analysis(save_path='tb_clustering_analysis.png')

        print("\n" + "="*50)
        print("EDA REPORT SUMMARY")
        print("="*50)
        print("📊 Comprehensive analysis completed!")
        print("📈 Distribution plots: tb_eda_distributions.png")
        print("🎯 Clustering analysis: tb_clustering_analysis.png")
        print(f"🔬 Dataset: {len(self.data)} samples, {(self.data['TB'] == 1).mean():.1%} TB prevalence")
        print("📋 Key insights generated")
        print("✅ Ready for further ML modeling!")


if __name__ == "__main__":
    # Generate comprehensive EDA report
    eda = TBEDA()
    eda.generate_report()
