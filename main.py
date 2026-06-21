import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as stats

st.set_page_config(
    page_title="Socio-Economic Analytics App",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('happy.csv')
    df['year'] = pd.to_numeric(df['year'])
    df1 = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()

    ncol1 = [
        'Life Ladder',
        'Social support',
        'Healthy life expectancy at birth',
        'Freedom to make life choices'
    ]
    happy_miss = df1[ncol1].isna().sum()
    raw_stats = df1[ncol1].describe()

    col1 = [
        'Country name',
        'year',
        'Life Ladder',
        'Social support',
        'Healthy life expectancy at birth',
        'Freedom to make life choices'
    ]
    df1 = df1[col1]
    df1 = df1.dropna()

    correct_columns = [
        'country_name', 'country_id', 'year', 'Inflation (CPI %)',
        'GDP (Current USD)', 'GDP per Capita (Current USD)', 'Unemployment Rate (%)',
        'Interest Rate (Real, %)', 'Inflation (GDP Deflator, %)', 'GDP Growth (% Annual)',
        'Current Account Balance (% GDP)', 'Government Expense (% of GDP)',
        'Government Revenue (% of GDP)', 'Tax Revenue (% of GDP)',
        'Gross National Income (USD)', 'Public Debt (% of GDP)'
    ]

    dff = pd.read_csv('ecomic.csv', skiprows=1, names=correct_columns, on_bad_lines='skip')
    dff['year'] = pd.to_numeric(dff['year'])
    df2 = dff[(dff['year'] >= 2021) & (dff['year'] <= 2023)].copy()

    ncol2 = [
        'Inflation (CPI %)',
        'GDP per Capita (Current USD)',
        'Unemployment Rate (%)',
        'GDP Growth (% Annual)'
    ]
    eco_miss = df2[ncol2].isna().sum()
    stats_raw = df2[ncol2].describe()

    col2 = [
        'country_name',
        'year',
        'Inflation (CPI %)',
        'GDP per Capita (Current USD)',
        'Unemployment Rate (%)',
        'GDP Growth (% Annual)'
    ]
    df2 = df2[col2]
    df2 = df2.dropna()

    df1.rename(columns={'Country name': 'country_name'}, inplace=True)
    df1['country_clean'] = df1['country_name'].astype(str).str.strip().str.lower()
    df2['country_clean'] = df2['country_name'].astype(str).str.strip().str.lower()

    df1['year'] = df1['year'].astype(int)
    df2['year'] = df2['year'].astype(int)

    country_rules = {
        "cote d'ivoire": "ivory coast",
        'kyrgyz republic': 'kyrgyzstan',
        'lao pdr': 'laos',
        'russian federation': 'russia',
        'slovak republic': 'slovakia',
        'turkiye': 'türkiye',
        'viet nam': 'vietnam'
    }

    df2['country_clean'] = df2['country_clean'].replace(country_rules)
    df_merged = pd.merge(df1, df2, on=['country_clean', 'year'], how='inner')

    if not df_merged.empty:
        df_merged['country_name'] = df_merged['country_name_x']
        df_merged = df_merged.drop(columns=['country_clean', 'country_name_x', 'country_name_y'], errors='ignore')

    if not df_merged.empty:
        df_merged['gdp_group'] = pd.qcut(df_merged['GDP per Capita (Current USD)'], q=3, labels=['Low GDP', 'Average GDP', 'High GDP'])

    return df_merged, happy_miss, raw_stats, eco_miss, stats_raw


df_merged, happy_miss, raw_stats, eco_miss, stats_raw = load_and_process_data()

st.sidebar.title("Navigation Panel")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Analysis Steps:",
    [
        "Abstract & Annotation",
        "Working with separate databases (2021-2023)",
        "Merging & Description statistics",
        "Visualizations",
        "Data transformation",
        "Hypothesis testing",
        "Country Explorer"
    ]
)

st.sidebar.markdown("---")

if page == "Working with separate databases (2021-2023)":
    selected_year = st.sidebar.slider("Temporal Filter (Year):", 2021, 2023, 2023)
    df_year = df_merged[df_merged['year'] == selected_year].copy()

if page == "Abstract & Annotation":
    st.title("1. Abstract / Annotation")
    st.markdown("""
    The aim of this work is to explore the relationship between national happiness levels and key economic indicators across countries for the period 2021–2023. The project analyzes how macro-financial variables like GDP per capita, inflation, and unemployment interact with subjective life satisfaction.

    **Team Contributions & Roles:**

    * **Enina Kira:**
      * Handled the initial data quality reporting and full cleanup process for the World Happiness Data 2024 | Emotions Analysis dataset (`happy.csv`).
      * Extracted core social metrics and calculated initial descriptive statistics (mean, median, and standard deviation) for the happiness parameters.
      * Generated baseline visualizations (histograms and dependencies) for individual social indicators.
      * Designed and developed a custom Telegram Bot interface for project presentation and quick data access.

    * **Kozlova Elizaveta:**
      * Managed custom parsing, column remapping, and full cleanup process for Global Economic Indicators (2010–2025)- World bank dataset (`ecomic.csv`).
      * Extracted core economic metrics and calculated initial descriptive statistics (mean, median, and standard deviation) for the economic parameters.
      * Generated baseline visualizations (histograms and dependencies) for individual economic indicators.
      * Engineered the data integration strategy, performing a multi-table `merge` based on normalized country keys and resolving naming anomalies.

    * **Collaborative work:**
      * Creation of graphs for the merged table. 
      * Testing of scientific hypotheses.
      * Preparation of the discussion sections.
      * Implementation of Streamlit.
    """)
    st.title("2. Dataset Description")
    st.markdown("""
        ### 2.1 Subject Area and Sourcing
        This project is situated within the domain of Socio-Economics and Data Analytics. To build a powerful and resilient analytical model, we constructed a composite dataset by manually integrating data from two independent reporting platforms:
        1. **Social Metrics (`happy.csv`):** Sourced from the annual World Happiness Report database, measuring psychological indicators of public well-being.
        2. **Economic Metrics (`ecomic.csv`):** Extracted from international financial statistics (World Bank parameters), tracking global economic stability and growth.

        The study restricts its data footprint to the timeframe of **2021–2023** to map recent worldwide macroeconomic conditions.

        ### 2.2 Fields and Initial Data Quality Report
        The combined dataset architecture observes 2 structural keys and 8 numerical parameters:
        * `Country name` / `country_name` (String) & `year` (Integer) - Used as structural merge alignment fields.
        * `Life Ladder` (Float) - Subjective happiness level on a scale from 0 to 10.
        * `Social support` & `Freedom to make life choices` (Floats) - Institutional safety nets and personal freedom scales.
        * `Healthy life expectancy at birth` (Float) - Projected life duration index.
        * `Inflation (CPI %)` & `GDP Growth (% Annual)` (Floats) - Core indicators of market and financial stability.
        * `GDP per Capita (Current USD)` & `Unemployment Rate (%)` (Floats) - Measures of national wealth distribution and labor dynamics.

        *Data Quality Assessment:* During the initial data exploration, several critical issues were found. First, the `year` indicators were stored as strings instead of proper numbers. Second, the raw economic CSV contained metadata noise requiring the parameter `skiprows=1` to align columns. Finally, both datasets suffered from missing records (NaNs), especially for developing economies.

        *Problem we faced:* Our original research design included `Public Debt (% of GDP)` as a core feature. However, due to an excessive volume of missing records in that specific column for 2021–2023, we made a data-driven decision to pivot and replace it with `Unemployment Rate (%)`, which offers comprehensive data density.
        """)

elif page == "Working with separate databases (2021-2023)":
    st.title("Working with separate databases")
    tab_happy_info, tab_happy, tab_eco_info, tab_eco = st.tabs(["Discussion: Happy", "Data: Happiness", "Discussion: Economic", "Data: Economics"])

    with tab_happy:
        st.subheader("Happiness Dataset Analysis (`happy.csv`)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Missing Values Before Cleanup:**")
            st.dataframe(happy_miss, use_container_width=True)
        with c2:
            st.markdown("**Descriptive Statistics Before Cleanup:**")
            st.dataframe(raw_stats.T, use_container_width=True)

        st.markdown(f"#### Baseline Indicator Visualizations ({selected_year})")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df_year['Life Ladder'], bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Level of Life Ladder')
            ax.set_xlabel('Life Ladder (0-10)')
            ax.set_ylabel('Amount of countries')
            st.pyplot(fig)
        with col_g2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_year['Social support'], df_year['Life Ladder'], alpha=0.5, color='green')
            ax.set_title('Happiness vs Social Support')
            ax.set_xlabel('Social support')
            ax.set_ylabel('Life Ladder')
            st.pyplot(fig)
        with col_g3:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_year['Healthy life expectancy at birth'], df_year['Life Ladder'], alpha=0.5, color='red')
            ax.set_title('Happiness vs Length of Life')
            ax.set_xlabel('Life expectancy (years)')
            ax.set_ylabel('Life Ladder')
            st.pyplot(fig)
    with tab_happy_info:
        st.title("Data cleanup & Descriptive statistics")
        st.markdown("""
        The raw data audit for `happy.csv` revealed visible missing cells in social metrics, which we safely filtered out using listwise row deletion (`dropna`). Before cleanup, missing values threatened to skew our sample. Our post-cleanup descriptive statistics show that the global average happiness score (`Life Ladder`) rests around 5.5, indicating a balanced global distribution."""
        )
        st.title("Plots")
        st.markdown("""
        1. **Happiness Distribution Histogram:** The 'level of happiness' chart reveals a single-modal, bell-shaped distribution, proving that the majority of tracked nations score in the moderate 5.0–6.5 zone, with very few extreme outliers.
        2. **Happiness vs. Social Support Scatter Plot:** Displays a prominent, upward-sloping linear trend. This confirms that countries providing high institutional safety nets systematically report higher public life evaluations.
        3. **Happiness vs. Life Expectancy Scatter Plot:** Uncovers a dense, exponential positive correlation, indicating that physical health and infrastructure longevity are deeply tied to collective mental well-being."""
        )
    with tab_eco:
        st.subheader("Macroeconomic Dataset Analysis (`ecomic.csv`)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Missing Values Before Cleanup:**")
            st.dataframe(eco_miss, use_container_width=True)
        with c2:
            st.markdown("**Descriptive Statistics Before Cleanup:**")
            st.dataframe(stats_raw.T, use_container_width=True)

        st.markdown(f"#### Macroeconomic trends & Distributions ({selected_year})")
        col_ec1, col_ec2 = st.columns(2)
        with col_ec1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df_year['Inflation (CPI %)'], bins=30, range=(-5, 40), color='skyblue', edgecolor='black',
                    alpha=0.7)
            ax.set_title('Distribution of Global Inflation Rate')
            ax.set_xlabel('Inflation (CPI %)')
            ax.set_ylabel('Number of countries')
            st.pyplot(fig)
        with col_ec2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df_year['GDP Growth (% Annual)'], df_year['Inflation (CPI %)'], color='darkviolet',
                       edgecolor='white', linewidth=0.5, alpha=0.7, s=40)
            ax.set_title('GDP Growth vs Inflation')
            ax.set_xlabel('Annual GDP growth')
            ax.set_ylabel('Inflation (CPI %)')
            ax.set_xlim(-15, 25)
            ax.set_ylim(-2, 40)
            st.pyplot(fig)

        st.markdown("#### Annual GDP Growth Dynamics for Key Economies (2021-2023)")
        fig, ax = plt.subplots(figsize=(12, 4))
        selected_countries = ['United States', 'China', 'Germany', 'Japan', 'India']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for country, color in zip(selected_countries, colors):
            data = df_merged[df_merged['country_name'] == country].sort_values('year')
            ax.plot(
                data['year'],
                data['GDP Growth (% Annual)'],
                marker='o',
                linewidth=2,
                color=color,
                label=country
            )
        ax.set_title('Dynamics of Annual GDP Growth (2021-2023)', pad=15)
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP Growth (% Annual)')
        ax.set_xticks([2021, 2022, 2023])
        ax.legend(title='Countries', loc='upper right')
        st.pyplot(fig)

    with tab_eco_info:
        st.title("Data cleanup & Descriptive statistics")
        st.markdown("""
        We removed rows with missing values because incomplete data makes it impossible to calculate accurate statistics or plot countries on our graphs.

        By keeping only the rows where all economic indicators are fully present, we got a clear and reliable dataset. This allows us to see the real global trends without broken or half-empty records messing up our analysis.

        Moreover, we managed custom parsing, column remapping because we had a problem with commas in this dataset, which were used as separators and were also present in the column names, causing the columns to be divided incorrectly.
        """)
        st.title("Plots")
        st.markdown("""
        1. **Distribution of the global inflation rate (2021-2023) Histogram:** Heavily skewed to the right. TIn the vast majority of countries, inflation rates remain below 10%, but the long right tail indicates significant systemic price surges across specific vulnerable regions during the 2021–2023 economic cycle.
        2. **The relationship between GDP Growth and Inflation (CPI %) Scatter Plot:** Illustrates a highly dense cluster near the 0–5% growth mark. It mathematically isolates extreme values, showing that hyperinflation does not automatically guarantee or correlate with high annual economic growth.
        3. **The relationship between GDP Growth and Inflation (CPI %) Line Chart:** The multi-line visualization of time series effectively contrasts the major world powers (US, China, Germany, etc.). It demonstrates a sharp decline and stabilization of the economy after 2021, proving how macroeconomic shocks (such as coronavirus) have different effects on various national structures over time.
        """)

elif page == "Merging & Description statistics":
    st.title("Merging & Description statistics")

    tab_merge_info, tab_merge_data, tab_stats_info, tab_stats_data = st.tabs([
        "Discussion: Merging",
        "Data: Harmonized Dataset",
        "Discussion: Statistics",
        "Data: Descriptive Statistics"
    ])

    with tab_merge_info:
        st.title("Data Integration Strategy")
        st.markdown("""
        We performed a multi-table `merge` operation based on normalized keys: **Country name** and **year**. This unified table allows us to directly evaluate how a nation's economic environment influences its social indices.

        During alignment, we resolved naming anomalies between the datasets (e.g., standardizing text fields to lowercase, trimming whitespaces, and remapping country aliases like 'Russian Federation' to 'Russia') to prevent synthetic rows generation or data loss.
        """)

    with tab_merge_data:
        st.subheader("Harmonized Final Dataframe Analysis")

        m1, m2 = st.columns(2)
        m1.metric("Number of rows in the final table", f"{df_merged.shape[0]}")
        m2.metric("Unique countries successfully glued together", f"{df_merged['country_name'].nunique()}")

        st.markdown("**First Rows of the Harmonized Final Dataframe:**")
        st.dataframe(df_merged.head(10), use_container_width=True)

    with tab_stats_info:
        st.title("Statistical distribution analysis")
        st.markdown("""
        We calculated the baseline central tendency and dispersion metrics (Mean, Median, and Standard Deviation) for both social and economic segments across the filtered timeframe. 

        This step allows us to check for skewness in financial parameters like GDP per capita and verify that psychological indicators like `Life Ladder` observe regular bounds stable enough for parametric hypothesis mapping.
        """)

    with tab_stats_data:
        st.subheader("Summary feature distribution metrics (Full Panel 2021-2023)")

        stats_cols = [
            'Life Ladder',
            'Social support',
            'Healthy life expectancy at birth',
            'Freedom to make life choices',
            'Inflation (CPI %)',
            'GDP per Capita (Current USD)',
            'Unemployment Rate (%)',
            'GDP Growth (% Annual)'
        ]

        cols_layout = st.columns(4)
        for index, col in enumerate(stats_cols):
            with cols_layout[index % 4]:
                st.markdown(f"**{col}**")
                st.code(
                    f"Mean:   {df_merged[col].mean():.2f}\n"
                    f"Median: {df_merged[col].median():.2f}\n"
                    f"Std:    {df_merged[col].std():.2f}"
                )

elif page == "Visualizations":
    st.title("Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Macro-Social Correlation Matrix",
        "2. Happiness by GDP Groups",
        "3. GDP Annual Growth vs Social Support",
        "4. Top-10 Happiness vs GDP"
    ])
    stats_cols = [
        'Life Ladder', 'Social support', 'Healthy life expectancy at birth',
        'Freedom to make life choices', 'Inflation (CPI %)',
        'GDP per Capita (Current USD)', 'Unemployment Rate (%)', 'GDP Growth (% Annual)'
    ]

    with tab1:
        st.header("Macro-Social Correlation Matrix")
        st.markdown("""
        ### Why we built this plot:
        To understand how deeply economic parameters are related to psychological well-being, we need to look at all variables simultaneously. To do this we constructed a **Pearson Linear Correlation Heatmap Matrix**.
        
        Complete square matrix visualizes every variable intersection twice, providing a look at how our social dataset (`df1`) and macroeconomic dataset (`df2`) are correlated. The main diagonal filled with 1.00 scores proves that each parameter perfectly aligns with itself, while the non-diagonal cells allow us to compare the strength of different conditions.

        ### What the plot tells us:
        By comparing rows and columns, we can identify several important global ideas:
        1. **Main factors:** People's happiness (`Life Ladder`) has its strongest positive connections with `Social support` ($0.82$) and `Healthy life expectancy at birth` ($0.75$). Right behind them is `GDP per Capita (Current USD)` with a strong score of $0.66$. This comparison shows us a clear fact: long-term stability, good health infrastructure and steady national wealth are deeply connected to how happy people actually feel.
        2. **Difference in GDP per capita and GDP Growth (% Annual):** When we compare stable wealth (`GDP per capita`) against quick yearly growth, we see a huge difference. Annual `GDP Growth (% Annual)` has almost zero connection with happiness ($0.07$). This proves that a quick, short-term boost in the stock market or economy doesn't automatically make citizens happier. Instead, it is the long-term, accumulated economic stability that truly satisfies people.
        """)

        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sb.heatmap(
            df_merged[stats_cols].corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson correlation coefficient'},
            ax=ax1
        )
        ax1.set_title('Matrix of Linear Correlations: Happiness vs Macroeconomics (2021-2023)', pad=20, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig1)

    with tab2:
        st.header("Distribution of Happiness by GDP Groups")
        st.markdown("""
        ### Why we built this plot:
        While our general correlation analysis gave us a single global number, it can sometimes hide the real differences between various economic classes.
        
        ### What the plot tells us:
        1. **The median shift:** As we move from poor countries to rich countries, there is a clear increase in happiness levels. The median inside the `High GDP` box is significantly higher than the line in the `Low GDP` box.
        2. **Deviation reduction:** If you look at the `Low GDP` and `Average GDP` columns, the black dots are heavily scattered from top to bottom. This shows that in lower-income segments money is not everything - some poor countries still manage to achieve a relatively high level of happiness, while others fall to the bottom. However, in the `High GDP` group the box narrows down significantly and the cloud of points compresses at the top of the scale. This tells us that high GDP effectively protects populations from low psychological scores.
        """)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sb.boxplot(
            data=df_merged,
            x='gdp_group',
            y='Life Ladder',
            fliersize=0,
            ax=ax2
        )
        sb.stripplot(
            data=df_merged,
            x='gdp_group',
            y='Life Ladder',
            color='black',
            alpha=0.5,
            size=4,
            jitter=0.2,
            ax=ax2
        )
        ax2.set_title('Distribution of happiness levels by country groups with different GDPs (2021-2023)', weight='bold')
        ax2.set_xlabel('Economic segment of countries (by GDP per capita)')
        ax2.set_ylabel('Level of Life Ladder')
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.header("Comparing the impact of GDP Growth on Social Support")
        st.markdown("""
        ### Why we built this plot:
        To gain a deeper understanding of non-subjective variables we constructed a multi-panel regression analysis (`FacetGrid`).
        
        ### What the plot tells us:
        By comparing the slopes of the lines across the three panels, we can see a significant structural difference in the functioning of the global economy:
        1. **Low GDP Segment:** On the leftmost panel, the blue line shows a clear upward trend. In developing countries, short-term economic growth acts as a powerful driving force. When the economy expands, it quickly generates real resources that immediately strengthen social support systems.
        2. **Average GDP Segment:** In the middle panel, the regression line is almost completely smoothed out. This indicates that as countries reach middle-income status, yearly fluctuations in GDP growth slow down their direct impact on the country's social structure.
        3. **High GDP Segment:** On the rightmost panel, the pink line remains perfectly flat and stays locked at the very top of the scale (around $0.85$-$0.90$). This provides a brilliant comparative insight: wealthy nations already possess permanent, deeply rooted social safety nets. Because these systems are heavily institutionalized, short-term yearly ups and downs in market growth do not change the baseline level of social support available to their citizens.
        """)

        g = sb.FacetGrid(df_merged, col="gdp_group", hue="gdp_group", height=5, aspect=1)
        g.map(
            sb.regplot,
            "GDP Growth (% Annual)",
            "Social support",
            scatter_kws={'alpha': 0.5, 's': 25},
            line_kws={'linewidth': 2}
        )
        g.set(xlim=(-10, 15))
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("GDP Growth (% Annual)", "Social support")
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle('Comparing the impact of GDP Growth on Social Support in different segments (2021-2023)', fontsize=14, weight='bold')
        st.pyplot(g.fig)

    with tab4:
        st.header("Comparison of Top-10 Happiness vs. Top-10 GDP (2023)")
        st.markdown("""
        ### Why we built this plot:
        Instead of looking at average trends, this analysis highlights the extremes of our dataset strictly on the **2023** cycle.
        
        ### What the plot tells us:
        By analyzing the printouts and comparing the two side-by-side graphs, we can draw three important comparative conclusions:
        1. **Intersection analysis:** The code calculates a direct mathematical intersection (`&`) between the two top-10 sets. The results show that out of the entire global dataset, only a few countries (highlighted in bright **green** on both sides) manage to be both the richest and the happiest in the world. This clearly demonstrates that while wealth may help, it does not guarantee a leading position in public mood.
        2. **Efficiency anomalies:** Several very rich countries (the orange bars) are missing from the happiness list, while countries with great social security and stability (the skyblue bars) take the top spots. This clearly proves one thing: once a country becomes rich, simply accumulating more cash (nominal GDP) does not make its population happier. What actually matters is how the government spends that money - specifically, by investing it back into social support and taking care of its citizens.
        """)

        df_2023 = df_merged[df_merged['year'] == 2023].copy()

        top_happiness = df_2023.nlargest(10, 'Life Ladder')[['country_name', 'Life Ladder', 'GDP per Capita (Current USD)']]
        top_gdp = df_2023.nlargest(10, 'GDP per Capita (Current USD)')[['country_name', 'Life Ladder', 'GDP per Capita (Current USD)']]

        happy_countries = set(top_happiness['country_name'])
        gdp_countries = set(top_gdp['country_name'])
        intersection = happy_countries & gdp_countries

        st.write(f"**Intersection of lists:** {len(intersection)} countries")
        st.code(f"{list(intersection) if intersection else 'No intersection'}")

        fig4, axes4 = plt.subplots(1, 2, figsize=(18, 8))
        colors_happy = ['green' if country in gdp_countries else 'skyblue' for country in top_happiness['country_name']]
        axes4[0].barh(top_happiness['country_name'], top_happiness['Life Ladder'], color=colors_happy, edgecolor='black')
        axes4[0].set_xlabel('Life Ladder (Happiness)', fontsize=12)
        axes4[0].set_title('Top-10 by Happiness (2023)\nGreen = also in GDP top-10', fontsize=12)
        axes4[0].invert_yaxis()

        for i, (idx, row) in enumerate(top_happiness.iterrows()):
            axes4[0].text(row['Life Ladder'] + 0.1, i, f"{row['Life Ladder']:.1f}", va='center')

        colors_gdp = ['green' if country in happy_countries else 'orange' for country in top_gdp['country_name']]
        axes4[1].barh(top_gdp['country_name'], top_gdp['GDP per Capita (Current USD)'] / 1000, color=colors_gdp, edgecolor='black')
        axes4[1].set_xlabel('GDP per Capita (thousand USD)', fontsize=12)
        axes4[1].set_title('Top-10 by GDP per Capita (2023)\nGreen = also in happiness top-10', fontsize=12)
        axes4[1].invert_yaxis()

        for i, (idx, row) in enumerate(top_gdp.iterrows()):
            axes4[1].text(row['GDP per Capita (Current USD)'] / 1000 + 0.5, i, f"${row['GDP per Capita (Current USD)'] / 1000:.0f}k", va='center')

        plt.suptitle('Comparison: Happiness vs GDP (2023)', fontsize=14, weight='bold')
        plt.tight_layout()
        st.pyplot(fig4)

elif page == "Data transformation":
    st.title("Data transformation")

    tab_discussion, tab_plots_data = st.tabs([
        "Discussion & Analysis",
        "Data & Visualizations"
    ])

    with tab_discussion:
        st.markdown("""
        # Data Transformation & Macroeconomic Shocks

        ### 1. Data Transformation
        To perform a deeper socio-economic shock analysis, we transformed the raw variables and constructed **4** new target metrics:
        * **`freedom_norm` & `unemploy_norm`:** We scaled `Freedom to make life choices` and `Unemployment Rate (%)` into a strict $[0, 1]$ range using a custom normalization function to eliminate scale disparities. They were constructed using standard Min-Max normalization:
           $$\\text{Normalized } X = \\frac{X - X_{min}}{X_{max} - X_{min} + 0.01}$$
        * **`efficiency` (Freedom-Employment efficiency index):** We constructed a synthetic indicator to evaluate how efficiently a country translates public freedom into labor market stability, calculated as:
           $$\\text{Efficiency} = \\frac{\\text{Normalized Freedom}}{\\text{Normalized Unemployment} + 0.01}$$
           This indicator evaluates whether a country is able to maintain the personal freedom of its citizens in the face of unemployment and economic difficulties. A higher value indicates a healthy economy where high personal freedom coexists with low unemployment.

        3. **`crisis` (Stagflation Crisis):**
           A custom boolean subset indicator that becomes `True` only when a country experiences simultaneous high inflation and stagnating growth (Stagflation):
           $$\\text{Crisis} = (\\text{Inflation (CPI \%)} > 8\\%) \\ \& \\ (\\text{GDP Growth (\% Annual)} < 1\\%)$$

        ### 2. Analysis of printed data subsets and statistics

        #### The efficiency leaders (Top 5)
        Our printed table highlights a very specific subgroup of anomalies, dominated by countries like **Cambodia** and **Niger**. These nations are the top efficiency spots because they report exceptionally low nominal unemployment rates (around $0.2\\text{-}0.4\\%$) while maintaining high levels of life choices. This demonstrates that the structural characteristics of the labor market can have a significant impact on our efficiency index beyond standard conditions of high GDP levels.

        #### The crisis subset performance
        The output filters and isolates countries experiencing severe stagflation. It presents various economic examples, ranging from extreme inflationary shocks such as **Lebanon** (inflation exceeding $150\\%$) to economies like **Czechia**, **Estonia**, and **Hungary** that are experiencing temporary post-pandemic or geopolitical growth declines combined with high consumer price indices.

        #### Definitive statistical contrast
        The final statistics clearly show how much an economic crisis hurts society:
        * **Average freedom decrease**: Stable nations score an average of **0.80** in life choices, while countries, experiencing a crisis, drop to  **0.73** (difference: **0.07 points**).
        * **Efficiency Squeeze**: The difference is even bigger when we look at our custom efficiency index. Stable countries have an average score of **6.76**, but in crisis zones, this drops to **5.80** (a loss of **0.96 points**). This clearly demonstrates that when a country experiences a crisis, it loses its ability to turn economic stability into real personal freedom for its citizens.

        ---

        ### 3. Graphic detailed overview: Freedom of Choice vs. Inflation
        * **The stable zone with green dots:** Stable countries cluster tightly on the left side where inflation is low (mostly between $0\\%$ and $6\\%$). Most of these green dots are located very high on the graph, which means that people there enjoy a lot of personal freedom (around $0.9$).
        * **The stagflation cluster:** The red dots move push to the right side of the graph as soon as inflation crosses the $8\\%$ mark. If you look closely at these red dots, you will see that they begin to move downward: the higher the inflation on the X-axis, the lower the freedom index on the Y-axis.
        * **The economic squeeze effect:** When the economy faces the problem of stagflation, rising prices make life difficult. The graph clearly shows that citizens in such conditions lose the ability to make choices, causing the red dots of the crisis to shift to lower positions on the graph.
        """)

    with tab_plots_data:
        st.header("Transformed Data & Visualizations")

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 0.01)

        df_merged['freedom_norm'] = normalize(df_merged['Freedom to make life choices'])
        df_merged['unemploy_norm'] = normalize(df_merged['Unemployment Rate (%)'])
        df_merged['efficiency'] = df_merged['freedom_norm'] / (df_merged['unemploy_norm'] + 0.01)
        df_merged['crisis'] = ((df_merged['Inflation (CPI %)'] > 8) & (df_merged['GDP Growth (% Annual)'] < 1))

        col_tables1, col_tables2 = st.columns(2)

        with col_tables1:
            st.markdown("#### TOP 5 COUNTRIES BY FREEDOM-EMPLOYMENT EFFICIENCY:")
            top5_eff = df_merged.nlargest(5, 'efficiency')[
                ['country_name', 'year', 'Freedom to make life choices', 'Unemployment Rate (%)', 'efficiency']]
            st.dataframe(top5_eff, use_container_width=True)

        with col_tables2:
            st.markdown("#### CRISIS COUNTRIES SUBSET PERFORMANCE:")
            crisis_df = df_merged[df_merged['crisis']]
            if len(crisis_df) > 0:
                st.dataframe(
                    crisis_df[['country_name', 'year', 'Inflation (CPI %)', 'GDP Growth (% Annual)',
                               'Freedom to make life choices']],
                    use_container_width=True
                )
            else:
                st.info("There are no countries in crisis")

        st.markdown("---")
        st.markdown("#### SUBSET COMPARISON (STABLE VS CRISIS)")

        crisis_freedom = df_merged[df_merged['crisis']]['Freedom to make life choices'].mean()
        stable_freedom = df_merged[~df_merged['crisis']]['Freedom to make life choices'].mean()

        crisis_eff = df_merged[df_merged['crisis']]['efficiency'].mean()
        stable_eff = df_merged[~df_merged['crisis']]['efficiency'].mean()

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Average Freedom to Make Life Choices**")
            st.write(f"Stable: `{stable_freedom:.2f}` | Crisis: `{crisis_freedom:.2f}`")
            st.metric(label="Freedom Difference (Stable - Crisis)", value=f"{stable_freedom - crisis_freedom:.2f}")

        with col_m2:
            st.markdown("**Average Freedom-Employment Efficiency Index**")
            st.write(f"Stable: `{stable_eff:.2f}` | Crisis: `{crisis_eff:.2f}`")
            st.metric(label="Efficiency Index Difference (Stable - Crisis)", value=f"{stable_eff - crisis_eff:.2f}")

        st.markdown("---")
        st.markdown("#### Visual Distribution Vector")
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        colors = df_merged['crisis'].map({True: 'red', False: 'green'})

        ax2.scatter(
            df_merged['Inflation (CPI %)'],
            df_merged['Freedom to make life choices'],
            c=colors,
            alpha=0.6,
            s=65,
            edgecolors='white',
            linewidths=0.5
        )

        ax2.set_xlim(-2, 30)
        ax2.set_xlabel('Inflation (CPI %)')
        ax2.set_ylabel('Freedom to Make Life Choices')
        ax2.set_title('Freedom of Choice vs Inflation (Red = segment of the artificially created stagflation crisis)',
                      pad=15, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)
elif page == "Hypothesis testing":
    st.title("Statistical Hypothesis Verification")
    tab_hyp1, tab_hyp2 = st.tabs([
        "1. GDP & Easterlin Paradox",
        "2. Unemployment vs Inflation"
    ])

    def p_value(x, y):
        n = len(x)
        if n <= 2:
            return 0.0, 1.0
        corr = np.corrcoef(x, y)[0, 1]
        if abs(corr) >= 1.0:
            return corr, 0.0
        t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
        return corr, p_val

    with tab_hyp1:
        st.markdown("""
        ### Hypothesis 1
        GDP per capita acts as a main basis for subjective public well-being, demonstrating a non-linear relationship with diminishing returns as wealth increases (The Easterlin Paradox).
        * **Verification status:** CONFIRMED.
        * **Statistical evidence:** The analysis resulted in a global Pearson correlation coefficient of $0.66$ with a statistical significance (p-value) of $0.00000$. Since the p-value is significantly lower than $0.05$ and the correlation exceeds $0.6$, the global hypothesis is strictly supported.
        * **Statistical interpretation & Key findings:** To evaluate the dynamic behind the *Easterlin Paradox*, we applied a logarithmic transformation (`logx=True`). While the global correlation of **$0.66$** supports a strong overall dependency, the non-linear trend line and segment behaviors leads to 2 critical insights:
          * **The law of diminishing returns:** In low-income countries, the logarithmic curve rises sharply. This clearly demonstrates how initial economic growth leads to an intense and immediate increase in social welfare through the satisfaction of basic needs (healthcare, safety, education). As countries transition to higher levels of GDP, the curve noticeably flattens, indicating that the absolute return of wealth on subjective happiness decreases significantly once a high standard of living is established.
          * **Segment variability vs Slope:** Within specific wealth levels, the correlation changes from $0.36$ (`Low GDP`) to $0.44$ (`Average GDP`) and $0.57$ (`High GDP`). This upward shift in coefficients does not mean money becomes more effective in rich nations. It reflects lower data volatility. In developing countries life ladder scores are highly volatile due to local political or social instability, resulting in a spread of scores. In high-income countries basic lifestyle indicators are highly stable and homogenous, resulting in the data points being very close to the flat trend line.
        """)

        st.markdown("---")
        st.subheader("Execution & Statistical Output")

        x1 = df_merged['GDP per Capita (Current USD)'].values
        y1 = df_merged['Life Ladder'].values
        corr1, p_val1 = p_value(x1, y1)

        st.write(f"**Pearson correlation coefficient:** `{corr1:.2f}`")
        st.write(f"**Statistical significance (p-value):** `{p_val1:.5f}`")

        if p_val1 < 0.05 and corr1 > 0.6:
            st.success(
                "Result: The hypothesis is CONFIRMED. The correlation is statistically significant and there is a strong positive relationship.")
        else:
            st.error(
                "Result: The hypothesis is REJECTED. The correlation could be accidental or the connection is weaker than expected.")

        st.markdown("**Segment analysis (Easterlin Paradox):**")
        for group in ['Low GDP', 'Average GDP', 'High GDP']:
            subset = df_merged[df_merged['gdp_group'] == group]
            if len(subset) > 2:
                sub_x = subset['GDP per Capita (Current USD)'].values
                sub_y = subset['Life Ladder'].values
                sub_corr, sub_p = p_value(sub_x, sub_y)
                st.write(f" * `{group}`: Correlation = `{sub_corr:.2f}`, p-value = `{sub_p:.5f}`")
            else:
                st.write(f" * `{group}`: Not enough data to calculate")

        fig1, ax1 = plt.subplots(figsize=(12, 7))
        colors1 = {'Low GDP': 'red', 'Average GDP': 'yellow', 'High GDP': 'green'}

        for group, color in colors1.items():
            subset = df_merged[df_merged['gdp_group'] == group]
            ax1.scatter(
                subset['GDP per Capita (Current USD)'],
                subset['Life Ladder'],
                label=group,
                color=color,
                alpha=0.6,
                edgecolors='white',
                linewidths=0.5,
                s=55
            )

        sb.regplot(
            data=df_merged,
            x='GDP per Capita (Current USD)',
            y='Life Ladder',
            scatter=False,
            logx=True,
            line_kws={'color': 'skyblue', 'linewidth': 3, 'label': 'Logarithmic Trend'},
            ax=ax1
        )

        ax1.set_title('Hypothesis 1: GDP per capita vs Life Ladder', fontsize=14, pad=15, weight='bold')
        ax1.set_xlabel('GDP per Capita (Current USD)', fontsize=11)
        ax1.set_ylabel('Life Ladder', fontsize=11)
        ax1.legend(frameon=True, facecolor='white', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig1)

    with tab_hyp2:
        st.markdown("""
        ### Hypothesis 2
        High unemployment rates have a more significant impact on individual freedom of choice than inflation, as the lack of job opportunities directly deprive citizens' economic activity.
          * **Verification status:** CONFIRMED.
          * **Statistical evidence:** The analysis evaluated the connection between economic factors such as `Unemployment Rate (%)` and `Inflation (CPI %)` and the social factor `Freedom to make life choices`. The calculations showed a Pearson correlation of $-0.37$ (p-value: $0.00000$) for unemployment, compared to a correlation of $-0.33$ (p-value: $0.00000$) for inflation.
          * **Key findings:** The graphs clearly show how these economic problems reduce people's freedom to make life choices. While both factors have a statistically significant negative impact, unemployment acts as a more important constraint. Inflation devalues monetary assets and purchasing power, but unemployment completely removes income stability and career options, making it much harder for people to plan their lives and stay independent.
        """)

        st.markdown("---")
        st.subheader("Execution & Statistical Output")

        freedom_data = df_merged['Freedom to make life choices'].values
        unemployment_data = df_merged['Unemployment Rate (%)'].values
        inflation_data = df_merged['Inflation (CPI %)'].values

        h2_corr1, h2_p_val1 = p_value(unemployment_data, freedom_data)
        h2_corr2, h2_p_val2 = p_value(inflation_data, freedom_data)

        st.write(
            f'**The relationship between freedom and unemployment:** Correlation = `{h2_corr1:.2f}`, p-value = `{h2_p_val1:.5f}`')
        st.write(
            f'**The relationship between freedom and inflation:** Correlation = `{h2_corr2:.2f}`, p-value = `{h2_p_val2:.5f}`')

        if abs(h2_corr1) > abs(h2_corr2):
            strong = 'Unemployment'
            weak = "Inflation"
            if h2_p_val1 < 0.05:
                st.success("Result: Hypothesis is CONFIRMED.")
            else:
                st.warning("Result: Hypothesis is REJECTED due to statistical insignificance.")
        else:
            strong = "Inflation"
            weak = "Unemployment"
            st.error("Result: Hypothesis is REJECTED (Inflation turned out to be more important).")

        st.info(f"The **'{strong}'** has a stronger negative impact on freedom of choice than **'{weak}'**.")

        fig2, (ax_sub1, ax_sub2) = plt.subplots(1, 2, figsize=(13, 5))

        sb.regplot(
            data=df_merged,
            x='Unemployment Rate (%)',
            y='Freedom to make life choices',
            scatter_kws={'alpha': 0.5, 'color': 'yellow'},
            line_kws={'color': 'skyblue', 'linewidth': 1.5},
            ax=ax_sub1
        )
        ax_sub1.set_title(f'Freedom vs Unemployment (r: {h2_corr1:.2f}, p: {h2_p_val1:.4f})', fontsize=11)
        ax_sub1.set_xlabel('Unemployment (%)')
        ax_sub1.set_ylabel('Freedom to make life choices')
        ax_sub1.grid(True, linestyle='--', alpha=0.4)

        sb.regplot(
            data=df_merged,
            x='Inflation (CPI %)',
            y='Freedom to make life choices',
            scatter_kws={'alpha': 0.5, 'color': 'pink'},
            line_kws={'color': 'skyblue', 'linewidth': 1.5},
            ax=ax_sub2
        )
        ax_sub2.set_title(f'Freedom vs Inflation (r: {h2_corr2:.2f}, p: {h2_p_val2:.4f})', fontsize=11)
        ax_sub2.set_xlabel('Inflation (CPI %)')
        ax_sub2.set_ylabel('')
        ax_sub2.set_xlim(-5, 40)
        ax_sub2.grid(True, linestyle='--', alpha=0.4)

        plt.suptitle('Comparing the impact of unemployment and inflation on freedom to make life choices', fontsize=13, y=1.02, weight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
elif page == "Country Explorer":
    st.title("Country Explorer & Profile Analysis")
    st.caption("Select or type any country to extract its full socioeconomic profile and dynamic trends (2021-2023).")

    tab_profile, tab_explorer_info = st.tabs([
        "Profile & Dynamics",
        "How Country Explorer Works"
    ])

    with tab_profile:
        unique_countries = sorted(df_merged['country_name'].unique())

        selected_country = st.selectbox(
            "Type or select a country:",
            options=unique_countries,
            index=unique_countries.index("United States") if "United States" in unique_countries else 0
        )

        country_data = df_merged[df_merged['country_name'] == selected_country].sort_values('year')

        if country_data.empty:
            st.error(f"No data available for {selected_country}.")
        else:
            latest_data = country_data[country_data['year'] == 2023]
            if latest_data.empty:
                latest_data = country_data.iloc[[-1]]

            st.subheader(f"Current Profile: {selected_country} ({int(latest_data['year'].values[0])})")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Life Ladder (Happiness)", f"{latest_data['Life Ladder'].values[0]:.2f}")
            with m2:
                gdp_val = latest_data['GDP per Capita (Current USD)'].values[0]
                st.metric("GDP per Capita", f"${gdp_val:,.0f}" if not pd.isna(gdp_val) else "N/A")
            with m3:
                inf_val = latest_data['Inflation (CPI %)'].values[0]
                st.metric("Inflation Rate", f"{inf_val:.1f}%" if not pd.isna(inf_val) else "N/A")
            with m4:
                unemp_val = latest_data['Unemployment Rate (%)'].values[0]
                st.metric("Unemployment", f"{unemp_val:.1f}%" if not pd.isna(unemp_val) else "N/A")

            st.markdown("---")
            st.subheader(f"Historical Macroeconomic Dynamics for {selected_country}")

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(country_data['year'], country_data['Life Ladder'], marker='o', color='purple', linewidth=2.5)
            axes[0].set_title('Happiness Trend')
            axes[0].set_ylabel('Life Ladder Score')
            axes[0].set_xticks(country_data['year'].astype(int))
            axes[0].grid(True, linestyle='--', alpha=0.5)

            axes[1].plot(country_data['year'], country_data['GDP per Capita (Current USD)'], marker='s', color='green',
                         linewidth=2.5)
            axes[1].set_title('GDP per Capita Trend')
            axes[1].set_ylabel('USD (Current)')
            axes[1].set_xticks(country_data['year'].astype(int))
            axes[1].grid(True, linestyle='--', alpha=0.5)

            axes[2].plot(country_data['year'], country_data['Inflation (CPI %)'], marker='^', color='red', linewidth=2.5)
            axes[2].set_title('Inflation Rate Trend')
            axes[2].set_ylabel('CPI %')
            axes[2].set_xticks(country_data['year'].astype(int))
            axes[2].grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Summary Insight")
            gdp_group = latest_data['gdp_group'].values[0]
            st.write(f"**{selected_country}** belongs to the **{gdp_group}** group based on its economic performance. ")

            if len(country_data) > 1:
                first_happy = country_data['Life Ladder'].values[0]
                last_happy = country_data['Life Ladder'].values[-1]
                diff = last_happy - first_happy
                if diff > 0.1:
                    st.success(
                        f"**Positive development:** Subjective well-being in {selected_country} has shown a visible **upward trend** over the analyzed post-pandemic period.")
                elif diff < -0.1:
                    st.error(
                        f"**Negative shift:** The overall life satisfaction score has **dropped** significantly. This suggests the presence of underlying social or macroeconomic distress factors.")
                else:
                    st.info(
                        f"**Stability:** The level of happiness in {selected_country} remains relatively **stable and stagnant** with no major shocks reported between 2021 and 2023.")

    with tab_explorer_info:
        st.markdown("""
        ## Country Explorer & Profile Analysis

        This interactive module is designed for an in-depth study of the socio-economic profiles of specific nations. Instead of generalized global trends, users can isolate data for a single country and observe its individual development.

        ### How the Algorithm Works:

        1. **User Selection:** Using the `st.selectbox` dropdown menu, the user selects the country of interest. The list is automatically generated using only the countries that were successfully combined from both databases during the merging stage (`df_merged['country_name'].unique()`).
        2. **Filtering & Sorting:** The program extracts rows corresponding to the selected country from the main dataframe and strictly sorts them in chronological order using `.sort_values('year')` to ensure the correct plotting of time series (2021–2023).
        3. **Calculation of Current Metrics:** The system finds the latest available year (defaulting to 2023) and displays key indicators in the form of four visual `st.metric` cards:
           * Current level of happiness (`Life Ladder`)
           * Economic wealth (`GDP per Capita`)
           * Inflation rate (`Inflation Rate`)
           * Labor market situation (`Unemployment`)
        4. **Plotting Historical Charts:** Using the `matplotlib` library, three independent line charts are generated to show the dynamics of change: how the happiness index changed, whether wealth rose or fell, and how stable prices remained over the three-year period.
        5. **Automated Text Analytics:** In the final step, the module automatically calculates the mathematical difference (`diff`) between the happiness level at the beginning of the period (2021) and the end (2023), producing a textual conclusion:
           * If the level has grown by more than 0.1 — a green success notification about **positive post-pandemic development** is displayed.
           * If it has dropped by more than -0.1 — a red warning about a **socio-economic crisis** is displayed.
           * In all other cases, the system reports the **stability** of the indicators.
        """)