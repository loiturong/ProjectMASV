import streamlit as st

def main():
    st.title(":rainbow[Machine Vision] :gray[Application]")

def about():
    st.title(":rainbow[About] :gray[Application]")

if __name__ == "__main__":
    pg = st.navigation([main, about])
    pg.run()