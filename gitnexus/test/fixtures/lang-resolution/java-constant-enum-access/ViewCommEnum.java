public enum ViewCommEnum {
    DEP_TYPCD_101("101", "a"),
    DEP_TYPCD_102("102", "b"),
    DEP_TYPCD_103("103", "c") { void hook() {} };
    private final String code;
    private final String name;
    ViewCommEnum(String code, String name) { this.code = code; this.name = name; }
    public String getCode() { return code; }
    public String getName() { return name; }
}
