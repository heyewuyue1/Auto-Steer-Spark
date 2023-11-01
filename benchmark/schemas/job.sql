CREATE TABLE IF NOT EXISTS aka_name (
    id integer NOT NULL,
    person_id integer NOT NULL,
    name string,
    imdb_index string,
    name_pcode_cf string,
    name_pcode_nf string,
    surname_pcode string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/aka_name.csv';

CREATE TABLE IF NOT EXISTS aka_title (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    title string,
    imdb_index string,
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code string,
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/aka_title.csv';

CREATE TABLE IF NOT EXISTS cast_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note string,
    nr_order integer,
    role_id integer NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/cast_info.csv';

CREATE TABLE IF NOT EXISTS char_name (
    id integer NOT NULL,
    name string NOT NULL,
    imdb_index string,
    imdb_id integer,
    name_pcode_nf string,
    surname_pcode string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/char_name.csv';

CREATE TABLE IF NOT EXISTS comp_cast_type (
    id integer NOT NULL,
    kind string NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/comp_cast_type.csv';

CREATE TABLE IF NOT EXISTS company_name (
    id integer NOT NULL,
    name string NOT NULL,
    country_code string,
    imdb_id integer,
    name_pcode_nf string,
    name_pcode_sf string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/company_name.csv';

CREATE TABLE IF NOT EXISTS company_type (
    id integer NOT NULL,
    kind string
) USING CSV LOCATION '/home/hejiahao/imdb/company_type.csv';

CREATE TABLE IF NOT EXISTS complete_cast (
    id integer NOT NULL,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/complete_cast.csv';

CREATE TABLE IF NOT EXISTS info_type (
    id integer NOT NULL,
    info string NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/info_type.csv';

CREATE TABLE IF NOT EXISTS keyword (
    id integer NOT NULL,
    keyword string NOT NULL,
    phonetic_code string
) USING CSV LOCATION '/home/hejiahao/imdb/keyword.csv';

CREATE TABLE IF NOT EXISTS kind_type (
    id integer NOT NULL,
    kind string
) USING CSV LOCATION '/home/hejiahao/imdb/kind_type.csv';

CREATE TABLE IF NOT EXISTS link_type (
    id integer NOT NULL,
    link string NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/link_type.csv';

CREATE TABLE IF NOT EXISTS movie_companies (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note string
) USING CSV LOCATION '/home/hejiahao/imdb/movie_companies.csv';

CREATE TABLE IF NOT EXISTS movie_info_idx (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info string NOT NULL,
    note string
) USING CSV LOCATION '/home/hejiahao/imdb/movie_info_idx.csv';

CREATE TABLE IF NOT EXISTS movie_keyword (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/movie_keyword.csv';

CREATE TABLE IF NOT EXISTS movie_link (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/movie_link.csv';

CREATE TABLE IF NOT EXISTS name (
    id integer NOT NULL,
    name string NOT NULL,
    imdb_index string,
    imdb_id integer,
    gender string,
    name_pcode_cf string,
    name_pcode_nf string,
    surname_pcode string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/name.csv';

CREATE TABLE IF NOT EXISTS role_type (
    id integer NOT NULL,
    role string NOT NULL
) USING CSV LOCATION '/home/hejiahao/imdb/role_type.csv';

CREATE TABLE IF NOT EXISTS title (
    id integer NOT NULL,
    title string NOT NULL,
    imdb_index string,
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code string,
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years string,
    md5sum string
) USING CSV LOCATION '/home/hejiahao/imdb/title.csv';

CREATE TABLE IF NOT EXISTS movie_info (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info string NOT NULL,
    note string
) USING CSV LOCATION '/home/hejiahao/imdb/movie_info.csv';

CREATE TABLE IF NOT EXISTS person_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info string NOT NULL,
    note string
) USING CSV LOCATION '/home/hejiahao/imdb/person_info.csv';